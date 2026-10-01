#!/usr/bin/env python3
"""Extract and check the real Q6_K 6-bit code unpack source in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost. The
temporary manifest wires the real `src/format/quant.rs` module by `#[path]`; the only new code is
the typed forwarding wrapper that calls the real `q6_k_codes` body. No substitute algorithm is
extracted.

Lane 2 (`/tmp/opencode/apeiron-lane-2.lock`), not the default lane, so a concurrent run on the
default lock cannot starve or interleave with this gate.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
LANE = '/tmp/opencode/apeiron-lane-2.lock'
if __name__ == '__main__' and os.environ.get('Q6K_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','Q6K_IN_LANE=1',
        'Q6K_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (failure register: /tmp tmpfs headroom); only named receipts and
# logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/q6kcodes-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path('/home/socol/oc-tmp/q6kcodes-resource')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('Q6K_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('Q6K_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles. `src/format/quant.rs` imports nothing from the rest
# of the crate, so `quant` alone is the real closure of `q6_k_codes`; only `quant` varies with the
# control copy.
LIB_TEMPLATE = '''#[path = {quant}]
pub mod quant;

/// Typed forwarding wrapper: it fixes the array type and calls the real
/// `src/format/quant.rs` `q6_k_codes` body. It restates no formula.
pub fn word_q6_k_codes(blk: &[u8]) -> Result<[i8; quant::QK_K], quant::QuantError> {{
    quant::q6_k_codes(blk)
}}
'''

def lib_source(quant):
    """Absolute `#[path]` to the real module; only `quant` varies with the control copy."""
    return LIB_TEMPLATE.format(quant=json.dumps(str(quant)))

def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="q6_k_codes_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant-codes' and label != 'mutant-blocks'))
    llbc = project/'q6_k_codes_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_q6_k_codes','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'Q6KCodesSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text().replace(str(source), 'src/format/quant.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('Q6K_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage q6kCodesProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib Q6KCodes where\n  roots := #[`Q6KCodesSource,`Q6KCodesResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='q6kCodesProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

# Two real-source controls, each a single occurrence in `src/format/quant.rs`:
#   R1 `codes`    the per-position inner-loop advance (the transition the work measure counts), and
#   R2 `blocks`   the outer-loop guard (the number of 128-weight halves per super-block).
CONTROLS = [
    ('codes', '            out[128 * n + l + 96] = q4 as i8;\n            l += 1;\n',
              '            out[128 * n + l + 96] = q4 as i8;\n            l += 2;\n'),
    ('blocks', '    while n < 2 {\n        let qb64 = 64 * n;\n        let qb32 = 32 * n;\n',
               '    while n < 3 {\n        let qb64 = 64 * n;\n        let qb32 = 32 * n;\n'),
]

def control_red(text, label):
    """The mutated extraction must turn the proof red with an attributable diagnostic."""
    if 'Q6KCodesResource.lean:' not in text or not any(x in text for x in
            ['unsolved goals','type mismatch','Tactic','unsolved metavariables']):
        raise RuntimeError('Q6K_CONTROL_NOT_ATTRIBUTABLE_'+label+' '+text[-1800:])

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/format/quant.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/Q6KCodesSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('Q6K_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('Q6K_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/Q6KCodesResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('Q6K_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'Q6KCodesSource.lean');shutil.copyfile(proof,p/'Q6KCodesResource.lean')
    code,text=run(['lake','build','Q6KCodesResource'],p,'fresh-proof')
    if 'Q6K_CODES_KERNEL_OK' not in text:raise RuntimeError('Q6K_AUDIT_MISSING')
    if a.controls:
        original=source.read_text()
        mutant=SCRATCH/'quant.rs'
        for label,needle,replacement in CONTROLS:
            if original.count(needle)!=1:raise RuntimeError('Q6K_CONTROL_TARGET_MISSING_'+label)
            mutant.write_text(original.replace(needle,replacement))
            freshmut=extract(mutant,'mutant-'+label);shutil.copyfile(freshmut,p/'Q6KCodesSource.lean')
            code,text=run(['lake','build','Q6KCodesResource'],p,'mutant-'+label+'-proof',require=False)
            if code!=1:raise RuntimeError('Q6K_CONTROL_NOT_RED_'+label+' exit '+str(code)+'\n'+text[-1800:])
            control_red(text,label)
            print('Q6K_REAL_SOURCE_CONTROL_RED_'+label)
        mutant.write_text(original);restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('Q6K_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'Q6KCodesSource.lean');run(['lake','build','Q6KCodesResource'],p,'restored-proof')
        print('Q6K_REAL_SOURCE_CONTROL_RESTORED')
    print('Q6K_CODES_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/format/quant.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/Q6KCodes_gate.py',ROOT/'proof/resource/Q6KCodesSource.lean',ROOT/'proof/resource/Q6KCodesResource.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('Q6K_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; the work measure counts loop iterations and array writes under named premises, never cycles, wall-clock or GPU time'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)