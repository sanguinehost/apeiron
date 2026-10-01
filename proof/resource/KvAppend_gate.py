#!/usr/bin/env python3
"""Extract and check the real KV-append source in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.

Extraction root. `crate::word_kv_append` (`proof/extract/lib.rs:910-917`) forwards to the real
`KvState::append` of `src/engine/attn.rs:886-950`, so this gate extracts that body directly from
the real crate with `--start-from crate::engine::attn::KvState::append`: no wrapper, no restated
formula, and the same body the production root names. The production root itself cannot reach it
at this revision (`crate::engine::mixer` is not in its `engine` mirror and `attn::f32_core` does
not resolve under `#[path]`), so the gate records that build as a diagnostic instead of depending
on it. Every stage runs in a throwaway copy under `/home/socol/oc-tmp`; `src/` is never written.

Stages: proof-artifact obligation -> identity -> extract-root diagnostic -> cargo build -> charon ->
aeneas -> body presence -> `--prepare` write -> proof policy -> `lake build` -> `--controls`
mutation/restore. The first stage fails closed while `proof/resource/KvAppendResource.lean` is
absent, which it is: no committed revision exists to recover.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
LANE = '/tmp/opencode/apeiron-lane-1.lock'
# Registered scratch location (docs/AGENTS.md failure register, tmpfs row): disposable copies
# live off the tmpfs; only named receipts and logs are written under /tmp/opencode.
WORK = Path('/home/socol/oc-tmp/kvappend-resource')
LOGS = Path('/tmp/opencode/kvappend-resource')
# The Lean spelling of the Charon path `apeiron::engine::attn::KvState::append`: the generated
# `def` line must name that body. Checked on a `def` line, so an empty namespace or a dropped
# body fails closed instead of passing on the surrounding declarations.
BODY_RE = re.compile(r'^def [^\n]*append\b', re.M)
if __name__ == '__main__' and os.environ.get('KV_APPEND_IN_LANE') != '1':
    probe = open(LANE,'a')
    try:
        import fcntl
        fcntl.flock(probe, fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(probe, fcntl.LOCK_UN)
    except OSError:
        print('LANE_BUSY '+LANE); probe.close(); sys.exit(75)
    probe.close()
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','KV_APPEND_IN_LANE=1',
        'KV_APPEND_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
DIAGNOSTICS = {}
LOGS.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
# Prune finished copies here, where this gate holds the lane, so no other run of it is active.
for stale in WORK.glob('kvappend-*'):
    if stale.is_dir():
        shutil.rmtree(stale, ignore_errors=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='kvappend-', dir=WORK))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START),
               float(os.environ.get('KV_APPEND_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('KV_APPEND_BUDGET_EXHAUSTED')
    log = LOGS / (SCRATCH.name+'-'+label+'.log')
    begin = time.monotonic()
    with log.open('wb') as f:
        p = subprocess.Popen(argv, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ,'LEAN_NUM_THREADS':'1'}, start_new_session=True)
        try:
            p.wait(timeout=left)
        except BaseException:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); raise
    text = log.read_text(errors='replace')
    RECORDS.append(dict(label=label,argv=argv,cwd=str(cwd),exit=p.returncode,
                        seconds=round(time.monotonic()-begin,3),log=str(log)))
    if require and p.returncode:
        raise RuntimeError(f'{label}_REJECT {p.returncode}\n{text[-3500:]}')
    return p.returncode,text

def tree(label, mutant=False):
    """Throwaway copy of the real crate; `mutant` is the one-guard real-source mutation."""
    dest = SCRATCH/('tree-'+label)
    dest.mkdir()
    for f in ('Cargo.toml','Cargo.lock'):
        shutil.copyfile(ROOT/f, dest/f)
    for d in ('src','derive','fold-abi'):
        shutil.copytree(ROOT/d, dest/d, symlinks=True)
    if mutant:
        p = dest/'src/engine/attn.rs'
        text = p.read_text()
        needle = 'if pos >= self.cap'
        if text.count(needle) != 1:
            raise RuntimeError('KV_APPEND_CONTROL_TARGET_MISSING')
        p.write_text(text.replace(needle, 'if pos > self.cap'))
    return dest

def extract(label, mutant=False):
    """Extract the real `KvState::append` body and require it to reach the Lean output."""
    project = tree(label, mutant)
    run(['cargo','build','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],
        project, label+'-rust-build', require=(label != 'mutant'))
    # The emitted file name is the LLBC stem, so keep the stem label-free: every extraction
    # writes `KvAppendSource.lean` and only the enclosing directory differs per label.
    stage = SCRATCH/label
    stage.mkdir()
    llbc = stage/'KvAppendSource.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas',
         '--dest-file',str(llbc),'--start-from','crate::engine::attn::KvState::append',
         '--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')], ROOT, label+'-charon')
    out = stage/'lean'
    out.mkdir()
    code,log = run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(out),
                    '-warnings-as-errors','-no-progress-bar','-all-computable',str(llbc)],
                   ROOT, label+'-aeneas', require=False)
    generated = out/'KvAppendSource.lean'
    text = generated.read_text() if generated.is_file() else ''
    body = bool(BODY_RE.search(text)) and 'Ignoring the body' not in log
    if not body:
        raise RuntimeError('KV_APPEND_EXTRACT_REJECT: the pinned Aeneas did not emit the real '
                           f'{BODY_RE.pattern} body (aeneas exit {code})\n'
                           + '\n'.join(l for l in log.splitlines() if 'Error' in l or 'Source:' in l))
    generated.write_text(generated.read_text().replace(str(project), 'src'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('KV_APPEND_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p = SCRATCH/'lean'; p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text(
        'import Lake\nopen Lake DSL\npackage kvAppendProof where\n'
        '  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\n'
        'require aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n'
        '@[default_target] lean_lib KvAppend where\n  roots := #[`KvAppendSource,`KvAppendResource]\n')
    manifest = json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text())
    manifest['name'] = 'kvAppendProof'
    for dep in manifest['packages']:
        if dep['name'] == 'aeneas': dep['dir'] = str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps = p/'.lake/packages'; deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir():
        (deps/dep.name).symlink_to(dep.resolve(), target_is_directory=True)
    return p

def extract_root_diagnostic():
    """Record why the production extraction root cannot name this body at this revision."""
    code,text = run(['cargo','build','--locked','--offline','--lib',
                     '--manifest-path',str(ROOT/'proof/extract/Cargo.toml'),
                     '--target-dir',str(SCRATCH/'mirror-target')], ROOT, 'extract-root-build',
                    require=False)
    errors = [l for l in text.splitlines() if l.startswith('error')]
    DIAGNOSTICS['extract_root'] = dict(exit=code, errors=errors)
    if code:
        print('KV_APPEND_EXTRACT_ROOT_BROKEN')
        for l in errors: print('  '+l)

PROOF_ARTIFACT = ROOT/'proof/resource/KvAppendResource.lean'

def proof_artifact_guard():
    """Fail closed on the unproven hand-written resource before any extraction is paid for.

    `KvAppendResource.lean` has no generator: `--prepare` writes only `KvAppendSource.lean`. This
    script is untracked at 028c8aa and the resource is absent from every ref, from `git log --all`,
    from `--diff-filter=D`, from every worktree and from the registered scratch roots, so the audit
    marker `KV_APPEND_KERNEL_OK` this gate requires has no artifact to come from. The gate is kept:
    the obligation is open until the resource is written and this gate re-runs green.
    """
    if not PROOF_ARTIFACT.is_file():
        print('KV_APPEND_PROOF_ARTIFACT_MISSING '+str(PROOF_ARTIFACT))
        print('KV_APPEND_OPEN_OBLIGATION: this resource has no committed revision to recover; it '
              'must be written and audited (kernel marker KV_APPEND_KERNEL_OK) before this gate '
              'can pass. Unproven: the KvState::append refinement. No silent pass.')
        raise SystemExit(2)

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--prepare', action='store_true')
    args.add_argument('--check', action='store_true')
    args.add_argument('--controls', action='store_true')
    a = args.parse_args()
    # `--prepare` and `--check` need only the generated source; the resource is the audited proof.
    if not (a.prepare or a.check):
        proof_artifact_guard()
    run(['rustc','--version'], ROOT, 'rustc-version')
    run(['cargo','--version'], ROOT, 'cargo-version')
    run(['lean','--version'], ROOT, 'lean-version')
    extract_root_diagnostic()
    generated = extract('fresh')
    dest = ROOT/'proof/resource/KvAppendSource.lean'
    if a.prepare:
        shutil.copyfile(generated, dest); print('KV_APPEND_EXTRACTION_READY'); return
    if a.check:
        print('KV_APPEND_EXTRACTION_MATCHES' if generated.read_bytes() == dest.read_bytes()
              else 'KV_APPEND_EXTRACTION_DRIFT')
        return
    proof = PROOF_ARTIFACT
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0', proof.read_text()):
        raise RuntimeError('KV_APPEND_PROOF_POLICY_REJECT')
    p = prepare_lean()
    shutil.copyfile(dest, p/'KvAppendSource.lean')
    shutil.copyfile(proof, p/'KvAppendResource.lean')
    code,text = run(['lake','build','KvAppendResource'], p, 'fresh-proof')
    if 'KV_APPEND_KERNEL_OK' not in text:
        raise RuntimeError('KV_APPEND_AUDIT_MISSING')
    if a.controls:
        mutant = extract('mutant', mutant=True)
        shutil.copyfile(mutant, p/'KvAppendSource.lean')
        code,text = run(['lake','build','KvAppendResource'], p, 'mutant-proof', require=False)
        if code != 1 or 'KvAppendResource.lean:' not in text or not any(
                x in text for x in ['unsolved goals','type mismatch','Tactic','failed to prove']):
            raise RuntimeError('KV_APPEND_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('KV_APPEND_REAL_SOURCE_CONTROL_RED')
        restored = extract('restored')
        # Source-location comments differ between disposable source paths; compare executable text.
        strip = lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '', s, flags=re.S)
        if strip(restored.read_text()) != strip(dest.read_text()):
            raise RuntimeError('KV_APPEND_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest, p/'KvAppendSource.lean')
        run(['lake','build','KvAppendResource'], p, 'restored-proof')
        print('KV_APPEND_REAL_SOURCE_CONTROL_RESTORED')
    print('KV_APPEND_SOURCE_RESOURCE_OK')

if __name__ == '__main__':
    status = 1
    try:
        main(); status = 0
    except SystemExit as e:
        status = e.code if isinstance(e.code, int) else 1
        raise
    finally:
        source_paths = [TOOLS/'target/aeneas/charon-driver', TOOLS/'target/aeneas-model/lean/lean-toolchain',
                        TOOLS/'proof/dispatch/lean-toolchain', ROOT/'src/engine/attn.rs',
                        ROOT/'Cargo.toml', ROOT/'Cargo.lock', ROOT/'proof/extract/lib.rs',
                        ROOT/'proof/resource/KvAppend_gate.py', ROOT/'proof/resource/KvAppendSource.lean',
                        ROOT/'proof/resource/KvAppendResource.lean', TOOLS/'target/aeneas/charon',
                        TOOLS/'target/aeneas/aeneas', TOOLS/'proof/dispatch/lake-manifest.json']
        (LOGS/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(
            exit=status, seconds=round(time.monotonic()-START,3), budget=540,
            argv=[sys.executable,*sys.argv], lane=LANE, memory_max='12G', swap_max=0, threads=1,
            proof_artifact=str(PROOF_ARTIFACT), proof_artifact_present=PROOF_ARTIFACT.is_file(),
            open_obligation=None if PROOF_ARTIFACT.is_file() else
                'KvAppendResource.lean absent from the tree and from every git ref; the '
                'KvState::append refinement (kernel marker KV_APPEND_KERNEL_OK) is unproven',
            deadline=os.environ.get('KV_APPEND_DEADLINE'), records=RECORDS, diagnostics=DIAGNOSTICS,
            extraction_root='crate::engine::attn::KvState::append of the real src/engine/attn.rs',
            forwarded_by='proof/extract/lib.rs:910-917 crate::word_kv_append',
            sha256={str(p):digest(p) for p in source_paths if p.is_file()},
            trust='Aeneas/Charon, compiler, IO/process/dependencies; the scalar work measure counts '
                  'converted/checked elements, never cycles, GPU time or wall-clock'),
            indent=2)+'\n')
        shutil.rmtree(SCRATCH)
