#!/usr/bin/env python3
"""Extract and check the real per-entry relocation core in a bounded private build.

The forwarding wrapper is narrow: it declares only `scalar`, `cell`, `coord` and `table` by
`#[path]`, so the extracted closure is the real `src/map/table.rs` `Table::relocate_entry` with its
`src/map/table/*` children. The shared `proof/extract` unit is not used, because it does not
currently compile. The `#[path]` roots are the production sources; no substitute algorithm.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT / 'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
LANE = '/tmp/opencode/apeiron-lane-3.lock'
if __name__ == '__main__' and os.environ.get('RELOCATE_ENTRY_IN_LANE') != '1':
    deadline = str(time.time() + 535)
    os.execvp('timeout', ['timeout', '--kill-after=5', '540s', 'flock', '-w', '530', LANE,
        'systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=12G', '-p', 'MemorySwapMax=0',
        'timeout', '--kill-after=5', '525s', 'env', 'LEAN_NUM_THREADS=1', 'RELOCATE_ENTRY_IN_LANE=1',
        'RELOCATE_ENTRY_DEADLINE=' + deadline, sys.executable, str(Path(__file__).resolve()),
        *sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch and control copies live off the tmpfs; only named receipts and logs go to /tmp/opencode.
LOGS = Path('/tmp/opencode/relocate-entry-resource')
LOGS.mkdir(parents=True, exist_ok=True)
BASE = Path(tempfile.gettempdir()) / 'relocate-entry-resource'
BASE.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='gate-', dir=BASE))


def digest(path):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic() - START),
               float(os.environ.get('RELOCATE_ENTRY_DEADLINE', time.time() + 525)) - time.time() - 3)
    if left <= 0:
        raise RuntimeError('RELOCATE_ENTRY_BUDGET_EXHAUSTED')
    log = LOGS / (SCRATCH.name + '-' + label + '.log')
    begin = time.monotonic()
    with log.open('wb') as handle:
        proc = subprocess.Popen(argv, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                env={**os.environ, 'LEAN_NUM_THREADS': '1'}, start_new_session=True)
        try:
            proc.wait(timeout=left)
        except BaseException:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
    text = log.read_text(errors='replace')
    RECORDS.append(dict(label=label, argv=argv, cwd=str(cwd), exit=proc.returncode,
                        seconds=time.monotonic() - begin, log=str(log)))
    if require and proc.returncode:
        raise RuntimeError(f'{label}_REJECT {proc.returncode}\n{text[-3500:]}')
    return proc.returncode, text


WRAPPER = '''#[path = {scalar}]
pub mod scalar;
#[path = {cell}]
pub mod cell;
#[path = {coord}]
pub mod coord;
#[path = {table}]
pub mod table;
pub use cell::Sigma;

/// The carried-key word instance the extracted root is monomorphic at.
pub type WordStoredTable = table::Table<i64, i64, table::WordHome, 16, table::keyrep::Stored<i64>>;

/// The real production per-entry relocation core through the stored carrier.
///
/// Calls the real `Table::relocate_entry` body; it does not restate the flat/spill write.
pub fn word_stored_relocate_entry(
    mut state: WordStoredTable,
    plan: &table::RebuildPlan,
    mut reserved: table::RebuildReserved<table::keyrep::Stored<i64>, i64>,
    entry_at: usize,
) -> (
    WordStoredTable,
    table::RebuildReserved<table::keyrep::Stored<i64>, i64>,
) {{
    state.relocate_entry(plan, &mut reserved, entry_at);
    (state, reserved)
}}
'''


def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH / label
    project.mkdir()
    (project / 'Cargo.toml').write_text(
        '[package]\nname="relocate_entry_source"\nversion="0.0.0"\nedition="2021"\n'
        '[lib]\npath="lib.rs"\n[workspace]\n')
    (project / 'lib.rs').write_text(WRAPPER.format(**{
        name: json.dumps(str(source / (name + '.rs'))) for name in
        ('scalar', 'cell', 'coord', 'table')}))
    run(['cargo', 'generate-lockfile', '--offline'], project, label + '-lock')
    run(['cargo', 'test', '--locked', '--offline', '--lib', '--target-dir',
         str(SCRATCH / 'rust-target')], project, label + '-rust-tests',
        require=(label != 'mutant'))
    llbc = project / 'relocate_entry_source.llbc'
    run([str(TOOLS / 'target/aeneas/charon'), 'cargo', '--sysroot', 'default', '--preset', 'aeneas',
         '--dest-file', str(llbc), '--start-from', 'crate::word_stored_relocate_entry', '--',
         '--manifest-path', str(project / 'Cargo.toml'), '--locked', '--offline', '--lib',
         '--target-dir', str(SCRATCH / 'charon-target')], project, label + '-charon')
    run([str(TOOLS / 'target/aeneas/aeneas'), '-backend', 'lean', '-dest', str(project),
         '-warnings-as-errors', '-no-progress-bar', '-all-computable', str(llbc)],
        project, label + '-aeneas')
    generated = project / 'RelocateEntrySource.lean'
    # Charon records `Source:` relative to the given root; strip trailing blanks as regen.sh does,
    # then drop the modeled-external stubs the pinned Aeneas model supplies.
    generated.write_text(re.sub(r'[ \t]+$', '', generated.read_text(), flags=re.M))
    run([sys.executable, str(TOOLS / 'proof/strip-modeled-axioms.py'), str(generated)],
        ROOT, label + '-strip-axioms')
    # A disposable source root is named by its own absolute path; normalize it so a control run is
    # byte-identical to the fresh run and the restore check can compare whole files.
    generated.write_text(generated.read_text().replace(str(source), 'src'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('RELOCATE_ENTRY_UNMODELED_SOURCE')
    return generated


def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    package = SCRATCH / 'lean'
    package.mkdir()
    (package / 'lean-toolchain').write_bytes((TOOLS / 'proof/dispatch/lean-toolchain').read_bytes())
    (package / 'lakefile.lean').write_text(
        'import Lake\nopen Lake DSL\npackage relocateEntryProof where\n'
        '  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\n'
        'require aeneas from ' + json.dumps(str(TOOLS / 'target/aeneas-model/lean')) + '\n'
        '@[default_target] lean_lib RelocateEntry where\n'
        '  roots := #[`RelocateEntrySource,`RelocateEntryResource]\n')
    manifest = json.loads((TOOLS / 'proof/dispatch/lake-manifest.json').read_text())
    manifest['name'] = 'relocateEntryProof'
    for dep in manifest['packages']:
        if dep['name'] == 'aeneas':
            dep['dir'] = str(TOOLS / 'target/aeneas-model/lean')
    (package / 'lake-manifest.json').write_text(json.dumps(manifest))
    deps = package / '.lake/packages'
    deps.mkdir(parents=True)
    for dep in (TOOLS / 'proof/dispatch/.lake/packages').iterdir():
        (deps / dep.name).symlink_to(dep.resolve(), target_is_directory=True)
    return package


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    source = ROOT / 'src'
    run(['rustc', '--version'], ROOT, 'rustc-version')
    run(['cargo', '--version'], ROOT, 'cargo-version')
    run(['lean', '--version'], ROOT, 'lean-version')
    generated = extract(source, 'fresh')
    dest = ROOT / 'proof/resource/RelocateEntrySource.lean'
    if args.prepare:
        shutil.copyfile(generated, dest)
        print('RELOCATE_ENTRY_EXTRACTION_READY')
        return
    if generated.read_bytes() != dest.read_bytes():
        raise RuntimeError('RELOCATE_ENTRY_EXTRACTION_DRIFT')
    proof = ROOT / 'proof/resource/RelocateEntryResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0', proof.read_text()):
        raise RuntimeError('RELOCATE_ENTRY_PROOF_POLICY_REJECT')
    package = prepare_lean()
    shutil.copyfile(dest, package / 'RelocateEntrySource.lean')
    shutil.copyfile(proof, package / 'RelocateEntryResource.lean')
    _, text = run(['lake', 'build', 'RelocateEntryResource'], package, 'fresh-proof')
    if 'RELOCATE_ENTRY_KERNEL_OK' not in text or 'forbidden axioms' in text or 'error' in text:
        raise RuntimeError('RELOCATE_ENTRY_AUDIT_MISSING')
    if args.controls:
        # The real source tree is copied; only this disposable copy is mutated.
        mutant = SCRATCH / 'mutant-src'
        shutil.copytree(source, mutant, symlinks=True)
        target = mutant / 'table.rs'
        original = target.read_text()
        needle = ('            reserved.resident.slots[index] = FlatSlot {\n'
                  '                live: true,\n'
                  '                sigma: entry_sigma,\n')
        if original.count(needle) != 1:
            raise RuntimeError('RELOCATE_ENTRY_CONTROL_TARGET_MISSING')
        target.write_text(original.replace(needle, needle.replace('live: true,', 'live: false,')))
        fresh = extract(mutant, 'mutant')
        shutil.copyfile(fresh, package / 'RelocateEntrySource.lean')
        code, text = run(['lake', 'build', 'RelocateEntryResource'], package, 'mutant-proof',
                         require=False)
        # The marker is a `run_cmd`, so a red build still logs it: the red signal is the exit
        # status plus a diagnostic attributed to a line of the proof file.
        if (code != 1 or 'RelocateEntryResource.lean:' not in text
                or not any(mark in text for mark in
                           ['unsolved goals', 'type mismatch', 'Tactic', 'declaration uses'])):
            raise RuntimeError('RELOCATE_ENTRY_CONTROL_NOT_ATTRIBUTABLE ' + text[-1800:])
        print('RELOCATE_ENTRY_REAL_SOURCE_CONTROL_RED')
        target.write_text(original)
        restored = extract(mutant, 'restored')
        if restored.read_bytes() != dest.read_bytes():
            raise RuntimeError('RELOCATE_ENTRY_RESTORE_DRIFT')
        shutil.copyfile(dest, package / 'RelocateEntrySource.lean')
        _, text = run(['lake', 'build', 'RelocateEntryResource'], package, 'restored-proof')
        if ('RELOCATE_ENTRY_KERNEL_OK' not in text or 'forbidden axioms' in text
                or 'error' in text):
            raise RuntimeError('RELOCATE_ENTRY_RESTORE_AUDIT_MISSING')
        print('RELOCATE_ENTRY_REAL_SOURCE_CONTROL_RESTORED')
    print('RELOCATE_ENTRY_SOURCE_RESOURCE_OK')


if __name__ == '__main__':
    status = 1
    try:
        main()
        status = 0
    finally:
        source_paths = [TOOLS / 'target/aeneas/charon-driver', TOOLS / 'target/aeneas/charon',
                        TOOLS / 'target/aeneas/aeneas', TOOLS / 'target/aeneas-model/lean/lean-toolchain',
                        TOOLS / 'proof/dispatch/lean-toolchain', TOOLS / 'proof/dispatch/lake-manifest.json',
                        TOOLS / 'proof/strip-modeled-axioms.py', ROOT / 'src/map/table.rs',
                        ROOT / 'src/map/table/state.rs', ROOT / 'src/map/table/locator.rs',
                        ROOT / 'src/map/table/probe.rs', ROOT / 'proof/resource/RelocateEntry_gate.py',
                        ROOT / 'proof/resource/RelocateEntrySource.lean',
                        ROOT / 'proof/resource/RelocateEntryResource.lean']
        (LOGS / (SCRATCH.name + '-receipt.json')).write_text(json.dumps(dict(
            exit=status, seconds=time.monotonic() - START, budget=540,
            argv=[sys.executable, *sys.argv], lane=LANE, memory_max='12G', swap_max=0, threads=1,
            deadline=os.environ.get('RELOCATE_ENTRY_DEADLINE'), records=RECORDS,
            sha256={str(path): digest(path) for path in source_paths if path.is_file()},
            trust='Aeneas/Charon, compiler, IO/process/dependencies; the claim is a scalar '
                  'source/resource claim about one relocation and excludes GPU/hardware timing',
        ), indent=2) + '\n')
        shutil.rmtree(SCRATCH, ignore_errors=True)
