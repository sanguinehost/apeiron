#!/usr/bin/env python3
"""Extract and check the real `word_coord_place` / `coord::spread` bodies in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
The narrow forwarding wrapper calls the real `src/coord.rs` through `#[path]`; the shared
`proof/extract/lib.rs` unit is not used and not touched.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
if __name__ == '__main__' and os.environ.get('COORD_PLACE_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',
        '/tmp/opencode/apeiron-lane-3.lock',
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','COORD_PLACE_IN_LANE=1',
        'COORD_PLACE_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
OUT = Path('/tmp/opencode/coordplace-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=OUT))
# The real modules the extracted bodies reach. `src/coord.rs` pulls `src/coord/address.rs`,
# `src/coord/locator.rs` and `src/coord/locator_core.rs` through its own `#[path]` attributes, so
# they are the transitive dependency set of the target root and are digested with it.
SOURCES = [ROOT/'src/coord.rs', ROOT/'src/cell.rs', ROOT/'src/scalar.rs', ROOT/'src/table.rs']
DEPENDS = SOURCES + [ROOT/'src/coord/address.rs', ROOT/'src/coord/locator.rs',
                     ROOT/'src/coord/locator_core.rs']
# `src/table.rs` reaches its own `#[path]` children, so the mutant tree needs their real content
# too; they are digested so the receipt names every source the extraction read.
TABLE_CHILDREN = [q for q in sorted((ROOT/'src/table').glob('*.rs'))]

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START),
               float(os.environ.get('COORD_PLACE_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('COORD_PLACE_BUDGET_EXHAUSTED')
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

def wrapper(dirpath, sources):
    """A narrow forwarding wrapper: each module is the real file reached by `#[path]`, and the
    entry points are the real `proof/extract/lib.rs` bodies over the real `coord::Addr`.

    The two `pub use` lines are the crate-root re-exports `src/lib.rs:33` and the `coord::locator`
    constants `src/coord/locator.rs` re-exports; the real files read `crate::Sigma`,
    `crate::LEVELS` and `crate::MAX_LEVELS` through them. No module body is restated. The shared
    `proof/extract/lib.rs` unit is deliberately not included: it does not compile against the
    current source, and this gate must not inherit that drift."""
    dirpath.mkdir(parents=True, exist_ok=True)
    mods = '\n'.join(f'#[path = {json.dumps(str(p))}]\npub mod {p.stem};' for p in sources)
    (dirpath/'lib.rs').write_text(
        mods + '\n'
        'pub use cell::{Cell, Sigma};\n'
        'pub fn word_coord_place(address: coord::Addr) -> u64 { address.place }\n'
        'pub fn word_coord_spread(x: u64) -> coord::Addr { coord::spread(x) }\n')
    (dirpath/'Cargo.toml').write_text(
        '[package]\nname="coord_place_source"\nversion="0.0.0"\nedition="2021"\n[lib]\n'
        'path="lib.rs"\n[workspace]\n')

def extract(sources, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    wrapper(project, sources)
    run(['cargo','generate-lockfile','--offline'], project, label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],
        project, label+'-rust-tests', require=(label != 'mutant'))
    llbc = project/'coord_place_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas',
         '--dest-file',str(llbc),
         '--start-from','crate::word_coord_place,crate::word_coord_spread',
         '--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')], project, label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),
         '-warnings-as-errors','-no-progress-bar','-all-computable',str(llbc)],
        project, label+'-aeneas')
    generated = project/'CoordPlaceSource.lean'
    text = generated.read_text()
    for p in DEPENDS:
        text = text.replace(str(p), str(p.relative_to(ROOT)))
    # A disposable mutant tree carries absolute paths in the generated comments; map them back to
    # their repository-relative names so the two extractions are comparable.
    mutant_root = SCRATCH/'mutant-src'
    for q in sorted(mutant_root.glob('**/*.rs'), reverse=True):
        if q.name != 'lib.rs':
            text = text.replace(str(q), 'src/'+str(q.relative_to(mutant_root)))
    generated.write_text(text)
    if re.search(r'\b(axiom|sorry|admit)\b', text):
        raise RuntimeError('COORD_PLACE_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p = SCRATCH/'lean'; p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text(
        'import Lake\nopen Lake DSL\npackage coordPlaceProof where\n'
        '  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\n'
        'require aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n'
        '@[default_target] lean_lib CoordPlace where\n'
        '  roots := #[`CoordPlaceSource,`CoordPlaceResource]\n')
    manifest = json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text())
    manifest['name'] = 'coordPlaceProof'
    for dep in manifest['packages']:
        if dep['name'] == 'aeneas':
            dep['dir'] = str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps = p/'.lake/packages'; deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir():
        (deps/dep.name).symlink_to(dep.resolve(), target_is_directory=True)
    return p

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--prepare', action='store_true')
    args.add_argument('--check', action='store_true')
    args.add_argument('--controls', action='store_true')
    a = args.parse_args()
    run(['rustc','--version'], ROOT, 'rustc-version')
    run(['cargo','--version'], ROOT, 'cargo-version')
    run(['lean','--version'], ROOT, 'lean-version')
    generated = extract(SOURCES, 'fresh')
    dest = ROOT/'proof/resource/CoordPlaceSource.lean'
    if a.prepare:
        shutil.copyfile(generated, dest); print('COORD_PLACE_EXTRACTION_READY'); return
    if generated.read_bytes() != dest.read_bytes():
        raise RuntimeError('COORD_PLACE_EXTRACTION_DRIFT')
    proof = ROOT/'proof/resource/CoordPlaceResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0', proof.read_text()):
        raise RuntimeError('COORD_PLACE_PROOF_POLICY_REJECT')
    p = prepare_lean()
    shutil.copyfile(dest, p/'CoordPlaceSource.lean')
    shutil.copyfile(proof, p/'CoordPlaceResource.lean')
    code, text = run(['lake','build','CoordPlaceResource'], p, 'fresh-proof')
    if 'COORD_PLACE_KERNEL_OK' not in text:
        raise RuntimeError('COORD_PLACE_AUDIT_MISSING')
    if a.controls:
        # Real-source mutation: drop the rotate the real body performs on the product's high half,
        # so the extracted placement fold is `hi ^ lo` where the proof states `rotl32 hi ^ lo`. The
        # change is in the real `src/coord.rs` body, not in a copy of the extracted term.
        original = (ROOT/'src/coord.rs').read_text()
        needle = 'place: hi.rotate_left(32) ^ lo,'
        if original.count(needle) != 1:
            raise RuntimeError('COORD_PLACE_CONTROL_TARGET_MISSING')
        # The mutant tree keeps the real layout, so `src/coord.rs`'s own `#[path]` children resolve
        # and only the one body differs from the real source.
        mutant_dir = SCRATCH/'mutant-src'
        # The real layout, so every `#[path]` child of every real module resolves; only the one
        # body differs from the real source.
        for src in sorted(set(DEPENDS) | set(TABLE_CHILDREN)):
            target = mutant_dir/src.relative_to(ROOT/'src')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
        (mutant_dir/'coord.rs').write_text(
            original.replace(needle, 'place: hi ^ lo,'))
        mut_sources = [mutant_dir/p.name for p in SOURCES]
        freshmut = extract(mut_sources, 'mutant')
        shutil.copyfile(freshmut, p/'CoordPlaceSource.lean')
        code, text = run(['lake','build','CoordPlaceResource'], p, 'mutant-proof', require=False)
        # The control must fail on the proof's own content: a type mismatch, an unsolved goal or a failed
        # tactic at a `CoordPlaceResource.lean` line. Lean's diagnostics put the line reference and
        # the reason on separate lines, so a reason line is attributed to the nearest preceding
        # `CoordPlaceResource.lean:<line>:` reference. A timeout or an unknown constant alone is a
        # budget or naming failure, not the attributable red this control exists to produce.
        attributable = []
        anchor = None
        for line in text.splitlines():
            stripped = line.strip()
            reasons = ('Type mismatch', 'unsolved goals', 'Tactic failed', 'failed to')
            m = re.match(r'error: CoordPlaceResource\.lean:(\d+)(:\d+)?', stripped)
            if m:
                if any(k in stripped for k in reasons):
                    attributable.append(f'CoordPlaceResource.lean:{m.group(1)}: {stripped}')
                else:
                    anchor = m.group(1)
            elif anchor is not None and any(k in stripped for k in reasons):
                attributable.append(f'CoordPlaceResource.lean:{anchor}: {stripped}')
                anchor = None
        if code != 1 or not attributable:
            raise RuntimeError('COORD_PLACE_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('COORD_PLACE_REAL_SOURCE_CONTROL_RED')
        for line in attributable[:3]:
            print('  mutant diagnostic: '+line[:300])
        (mutant_dir/'coord.rs').write_text(original)
        restored = extract(mut_sources, 'restored')
        # Source-location comments differ for disposable absolute source paths; compare body text.
        strip = lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '', s, flags=re.S)
        if strip(restored.read_text()) != strip(generated.read_text()):
            raise RuntimeError('COORD_PLACE_RESTORE_BODY_DRIFT')
        if restored.read_bytes() != generated.read_bytes():
            raise RuntimeError('COORD_PLACE_RESTORE_DRIFT')
        shutil.copyfile(dest, p/'CoordPlaceSource.lean')
        run(['lake','build','CoordPlaceResource'], p, 'restored-proof')
        print('COORD_PLACE_REAL_SOURCE_CONTROL_RESTORED')
    print('COORD_PLACE_SOURCE_RESOURCE_OK')

if __name__ == '__main__':
    status = 1
    try:
        main(); status = 0
    finally:
        source_paths = sorted(set(DEPENDS) | set(TABLE_CHILDREN)) + [
            TOOLS/'target/aeneas/charon-driver',
            TOOLS/'target/aeneas-model/lean/lean-toolchain',
            TOOLS/'proof/dispatch/lean-toolchain',
            ROOT/'proof/resource/CoordPlace_gate.py',
            ROOT/'proof/resource/CoordPlaceSource.lean',
            ROOT/'proof/resource/CoordPlaceResource.lean',
            TOOLS/'target/aeneas/charon', TOOLS/'target/aeneas/aeneas',
            TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(
            exit=status, seconds=time.monotonic()-START, budget=540,
            argv=[sys.executable,*sys.argv], lane='/tmp/opencode/apeiron-lane-3.lock',
            memory_max='12G', swap_max=0, threads=1,
            deadline=os.environ.get('COORD_PLACE_DEADLINE'), records=RECORDS,
            sha256={str(q):digest(q) for q in source_paths if q.is_file()},
            trust=('Aeneas/Charon, the Rust compiler, the u128 wrapping_mul model, IO/process/'
                   'dependencies; the scalar source/resource claim excludes GPU, hardware timing '
                   'and instruction counts')), indent=2)+'\n')
        shutil.rmtree(SCRATCH)
