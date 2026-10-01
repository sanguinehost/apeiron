#!/usr/bin/env python3
"""Check the cost bridge in a bounded private Lean package.

The bridge is copied into a throwaway Lake package that resolves the pinned toolchain from
target/aeneas-model/lean; only the bridge copy is ever mutated. `CostAlgebra` and `RowDotResource`
each register an elaborator named `#resource_audit`, so no single file may import both; the gate
therefore builds verbatim copies of the two with that one identifier renamed, and refuses to run
unless each copy is byte-equal to the committed file modulo exactly that rename. Dependency
artifacts are read from the pinned lane, never rebuilt. O(B) module bytes are read per build;
external compilation is timed, not assigned constant cost. Each of the two seeded mapping defects
must turn the build red with an attributable diagnostic naming the headline it broke, and the
restored copy must build green.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'proof/resource/CostBridge.lean'
GATE = Path(__file__).resolve()
COST_ALGEBRA = ROOT / 'proof/resource/CostAlgebra.lean'
ROW_DOT_RESOURCE = ROOT / 'proof/resource/RowDotResource.lean'
ROW_DOT_SOURCE = ROOT / 'proof/resource/RowDotSource.lean'
AENEAS_MODEL = ROOT / 'target/aeneas-model/lean'
LANE = '/tmp/opencode/apeiron-lane-3.lock'
TOOLCHAIN = 'leanprover/lean4:v4.31.0'
MARKER = 'COST_BRIDGE_KERNEL_OK'
AUDIT = 'COST_BRIDGE_AUDIT'
REJECT = 'COST_BRIDGE_AXIOM_REJECT'
MODULE_IMPORTS = ('import CostAlgebra', 'import RowDotResource')
BUDGET = 540
INNER = 525

if __name__ == '__main__' and os.environ.get('COST_BRIDGE_IN_LANE') != '1':
    deadline = str(time.time() + (INNER + 5))
    os.execvp('timeout', ['timeout', '--kill-after=5', str(BUDGET) + 's', 'flock', '-w',
        str(INNER + 5), LANE, 'systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=12G',
        '-p', 'MemorySwapMax=0', 'timeout', '--kill-after=5', str(INNER) + 's', 'env',
        'LEAN_NUM_THREADS=1', 'COST_BRIDGE_IN_LANE=1', 'COST_BRIDGE_DEADLINE=' + deadline,
        sys.executable, str(GATE), *sys.argv[1:]])

START = time.monotonic()
RECORDS = []
OUT = Path('/tmp/opencode/cost-bridge')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path('/home/socol/oc-tmp/cost-bridge')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='gate-', dir=SCRATCH_ROOT))

# The single identifier renamed in each dependency copy, and why: two modules registering one
# elaborator name cannot be imported together.
RENAMES = (
    ('CostAlgebra.lean', COST_ALGEBRA, '#resource_audit ', '#cost_algebra_audit '),
    ('RowDotResource.lean', ROW_DOT_RESOURCE, '#resource_audit ', '#row_dot_resource_audit '),
)

# Each control: a single-needle text defect seeded in the throwaway bridge copy, the headlines that
# must go red with it, and the defect's name for the receipt. Both needles are the one line of
# `toCost`, so the two controls differ only in what they do to that line.
NEEDLE = '  ⟨p.mulAdds, p.bytes, p.allocs, 0, 0, 0, p.visits⟩'
CONTROLS = (
    ('dropped-multiply-add', '⟨0, p.bytes, p.allocs, 0, 0, 0, p.visits⟩',
     ('CostBridge.exact_run_cost', 'CostBridge.padded_run_cost', 'CostBridge.composed_exact_cost',
      'CostBridge.composed_padded_cost')),
    ('non-monotone-critical-path',
     '⟨p.mulAdds, p.bytes, p.allocs, 0, 0, 0, p.visits - p.mulAdds⟩',
     ('CostBridge.toCost_monotone_visits', 'CostBridge.exact_run_cost')),
)
ERROR_KINDS = ('unsolved goals', 'Application type mismatch', 'Type mismatch',
               'Tactic `decide` proved', "'show' tactic failed", 'Tactic `cases` failed',
               'does not synthesize', 'failed to synthesize')
DECL_RE = re.compile(r'^(?:@\[\w+\]\s*)?(?:theorem|def|abbrev|structure|inductive|instance)\b',
                     re.M)
LOCATION_RE = re.compile(r'(?:^|\s)(?:error:\s)?CostBridge\.lean:(\d+):(\d+): (.*)$', re.M)
FORBIDDEN_RE = re.compile(r'\b(sorry|axiom|admit)\b|set_option maxHeartbeats 0')
# Raised recursion or heartbeat budgets defeat the reduction guard. The committed extraction is
# exempt because its own generator writes them; this gate's module is not.
DEPTH_RE = re.compile(r'maxRecDepth|maxHeartbeats')


def digest(path):
    """O(B) bytes; B is one module copy."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def left():
    """Seconds left in the whole invocation, lane wait included."""
    return min(INNER - (time.monotonic() - START),
               float(os.environ.get('COST_BRIDGE_DEADLINE', time.time() + INNER)) - time.time() - 3)


def run(argv, cwd, label, require=True):
    """Run inside the shared budget and keep the complete output."""
    remaining = left()
    if remaining <= 0:
        raise RuntimeError('COST_BRIDGE_BUDGET_EXHAUSTED')
    log = OUT / (SCRATCH.name + '-' + label + '.log')
    begin = time.monotonic()
    with log.open('wb') as handle:
        proc = subprocess.Popen(argv, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True,
                                env={**os.environ, 'LEAN_NUM_THREADS': '1'})
        try:
            proc.wait(timeout=remaining)
        except BaseException:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
    text = log.read_text(errors='replace')
    RECORDS.append(dict(label=label, argv=[str(a) for a in argv], cwd=str(cwd),
                        exit=proc.returncode, seconds=round(time.monotonic() - begin, 3),
                        log=str(log)))
    if require and proc.returncode:
        raise RuntimeError(f'{label}_REJECT {proc.returncode}\n{text[-3000:]}')
    return proc.returncode, text


def renamed_copy(name, path, old, new):
    """The dependency copy with one identifier renamed, proved equal to the committed file."""
    original = path.read_text()
    if original.count(old) < 2:
        raise RuntimeError('COST_BRIDGE_RENAME_TARGET_MISSING ' + name)
    candidate = original.replace(old, new)
    if candidate.replace(new, old) != original:
        raise RuntimeError('COST_BRIDGE_RENAME_NOT_EXACT ' + name)
    return candidate


def preflight():
    """Pin the toolchain, the lane manifest and the module policy before any build."""
    toolchain = AENEAS_MODEL / 'lean-toolchain'
    if not toolchain.is_file() or toolchain.read_text().strip() != TOOLCHAIN:
        raise RuntimeError('COST_BRIDGE_TOOLCHAIN_REJECT')
    if not (ROOT / 'proof/dispatch/lean-toolchain').is_file():
        raise RuntimeError('COST_BRIDGE_LANE_MANIFEST_MISSING')
    if not (ROOT / 'proof/dispatch/lake-manifest.json').is_file():
        raise RuntimeError('COST_BRIDGE_LANE_DEPENDENCIES_MISSING')
    for path in (MODULE, COST_ALGEBRA, ROW_DOT_RESOURCE, ROW_DOT_SOURCE):
        if FORBIDDEN_RE.search(path.read_text()):
            raise RuntimeError('COST_BRIDGE_MODULE_POLICY_REJECT ' + str(path))
    if DEPTH_RE.search(MODULE.read_text()):
        raise RuntimeError('COST_BRIDGE_BUDGET_OPTION_REJECT ' + str(MODULE))
    module = MODULE.read_text()
    imports = [line for line in module.splitlines() if line.startswith('import ')]
    if tuple(imports) != MODULE_IMPORTS:
        raise RuntimeError('COST_BRIDGE_IMPORT_CONTRACT_DRIFT ' + repr(imports))
    if len(re.findall(r'^run_cmd Lean\.logInfo "' + MARKER + r'"$', module, re.M)) != 1:
        raise RuntimeError('COST_BRIDGE_MARKER_MISSING')
    if module.count(NEEDLE) != 1:
        raise RuntimeError('COST_BRIDGE_CONTROL_TARGET_MISSING')
    for _, path, old, new in RENAMES:
        renamed_copy(path.name, path, old, new)
    return module


def prepare(module):
    """Private package; dependency artifacts are read from the pinned lane, never rebuilt."""
    package = SCRATCH / 'lean'
    package.mkdir()
    (package / 'lean-toolchain').write_bytes((AENEAS_MODEL / 'lean-toolchain').read_bytes())
    (package / 'lakefile.lean').write_text(
        'import Lake\nopen Lake DSL\npackage costBridge where\n'
        '  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\n'
        'require aeneas from ' + json.dumps(str(AENEAS_MODEL)) + '\n'
        '@[default_target] lean_lib CostBridge where\n'
        '  roots := #[`CostAlgebra, `RowDotSource, `RowDotResource, `CostBridge]\n')
    manifest = json.loads((ROOT / 'proof/dispatch/lake-manifest.json').read_text())
    manifest['name'] = 'costBridge'
    for entry in manifest['packages']:
        if entry['name'] == 'aeneas':
            entry['dir'] = str(AENEAS_MODEL)
    (package / 'lake-manifest.json').write_text(json.dumps(manifest))
    packages = package / '.lake/packages'
    packages.mkdir(parents=True)
    for dependency in (ROOT / 'proof/dispatch/.lake/packages').iterdir():
        link = packages / dependency.name
        if not link.exists():
            link.symlink_to(dependency.resolve(), target_is_directory=True)
    shutil.copyfile(ROW_DOT_SOURCE, package / 'RowDotSource.lean')
    for name, path, old, new in RENAMES:
        (package / name).write_text(renamed_copy(name, path, old, new))
    clean_build_artifacts(package)
    (package / 'CostBridge.lean').write_text(module)
    return package


def clean_build_artifacts(package):
    """Drop this module's products so a mutated copy is certainly recompiled. The renamed
    dependency copies keep their oleans: only the bridge is mutated."""
    root = package / '.lake/build'
    for sub in ('lib/lean', 'ir'):
        directory = root / sub
        if not directory.is_dir():
            continue
        for path in directory.glob('CostBridge.*'):
            if path.is_file():
                path.unlink()


def build(package, text, label, require=True):
    """Install one bridge copy, build it, and return the complete log."""
    (package / 'CostBridge.lean').write_text(text)
    clean_build_artifacts(package)
    return run(['lake', 'build', 'CostBridge'], package, label, require=require)


def audited_headlines(module):
    """Every headline the module asks to have audited, in source order, fully qualified."""
    return re.findall(r'^#resource_audit ([\w.]+)$', module, re.M)


def declaration_span(module, headline):
    """The line range of one headline's own declaration, its docstring included. The headline may
    be qualified; only its last component names the declaration."""
    name = headline.split('.')[-1]
    match = re.search(r'^(?:theorem|def) ' + re.escape(name) + r'\b', module, re.M)
    if match is None:
        raise RuntimeError('COST_BRIDGE_TARGET_HEADLINE_MISSING ' + headline)
    start = module.count('\n', 0, match.start()) + 1
    following = [m.start() for m in DECL_RE.finditer(module) if m.start() > match.start()]
    end = module.count('\n', 0, following[0]) if following else module.count('\n')
    return start, end


def green(text, module, label):
    """A green build: nothing red, one audit per requested headline, one marker."""
    if REJECT in text or 'COST_ALGEBRA_AXIOM_REJECT' in text or 'forbidden axioms' in text:
        raise RuntimeError(label + '_AUDIT_REJECT\n' + text[-2000:])
    missing = [name for name in audited_headlines(module)
               if AUDIT + ' ' + name + ':' not in text]
    if missing:
        raise RuntimeError(label + '_AUDIT_MISSING ' + repr(missing) + '\n' + text[-2000:])
    if text.count(MARKER) != 1:
        raise RuntimeError(label + '_MARKER_MISSING\n' + text[-2000:])
    if 'error:' in text:
        raise RuntimeError(label + '_ERROR_PRESENT\n' + text[-2000:])
    return text


def diagnostic_lines(text):
    """Source lines carrying a real diagnostic, excluding the audit's own report lines, its reject
    lines and the completion marker, which all sit at column 0 on the `#resource_audit` lines."""
    skipped = (REJECT, AUDIT, MARKER)
    return sorted({int(match.group(1)) for match in LOCATION_RE.finditer(text)
                   if not match.group(3).startswith(skipped)})


def attributable(code, text, module, targets, label):
    """The seeded defect must name every headline it broke, and place a diagnostic in each one's
    own declaration. A control that fails elsewhere is not evidence about this module."""
    if code == 0:
        raise RuntimeError(label + '_CONTROL_GREEN\n' + text[-1500:])
    if 'error:' not in text:
        raise RuntimeError(label + '_CONTROL_NO_ERROR\n' + text[-1500:])
    locations = diagnostic_lines(text)
    if not locations:
        raise RuntimeError(label + '_CONTROL_UNATTRIBUTED\n' + text[-2000:])
    unnamed = [name for name in targets
               if REJECT + ' ' + name + ':' not in text]
    if unnamed:
        raise RuntimeError(label + '_CONTROL_HEADLINE_NOT_NAMED ' + repr(unnamed) + '\n'
                           + text[-2000:])
    out_of_span = [name for name in targets
                   if not any(declaration_span(module, name)[0] <= line
                              <= declaration_span(module, name)[1] for line in locations)]
    if out_of_span:
        raise RuntimeError(label + '_CONTROL_DIAGNOSTIC_OUTSIDE_HEADLINE ' + repr(out_of_span)
                           + ' locations=' + repr(locations) + '\n' + text[-2000:])
    if not any(kind in text for kind in ERROR_KINDS):
        raise RuntimeError(label + '_CONTROL_NO_DIAGNOSTIC\n' + text[-2000:])
    return locations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    module = preflight()
    run(['lake', '--version'], ROOT, 'lake-version')
    package = prepare(module)
    code, text = build(package, module, 'fresh-proof')
    green(text, module, 'FRESH')
    for marker in ('COST_ALGEBRA_KERNEL_OK', 'ROW_DOT_KERNEL_OK'):
        if marker not in text:
            raise RuntimeError('FRESH_RENAMED_DEPENDENCY_INCOMPLETE ' + marker + '\n'
                               + text[-2000:])
    headlines = audited_headlines(module)
    print('COST_BRIDGE_PROOF_OK check=%s headlines=%d lanes=%s'
          % (args.check, len(headlines), ','.join(name for name, _, _, _ in RENAMES)))
    if args.controls:
        for name, replacement, targets in CONTROLS:
            code, text = build(package, module.replace(NEEDLE, replacement),
                               'mutant-' + name, require=False)
            locations = attributable(code, text, module, targets,
                                     'COST_BRIDGE_' + name.upper().replace('-', '_'))
            print('COST_BRIDGE_CONTROL_RED %s targets=%s lines=%s'
                  % (name, ','.join(targets), locations))
            code, text = build(package, module, 'restored-' + name)
            green(text, module, 'RESTORED_' + name.upper().replace('-', '_'))
            print('COST_BRIDGE_CONTROL_RESTORED %s' % name)
    print('COST_BRIDGE_COST_MODEL_OK')


if __name__ == '__main__':
    status = 1
    try:
        main()
        status = 0
    finally:
        sources = [MODULE, GATE, COST_ALGEBRA, ROW_DOT_RESOURCE, ROW_DOT_SOURCE,
                   AENEAS_MODEL / 'lean-toolchain', ROOT / 'proof/dispatch/lake-manifest.json']
        (OUT / (SCRATCH.name + '-receipt.json')).write_text(json.dumps(dict(
            exit=status, seconds=round(time.monotonic() - START, 3), budget=BUDGET,
            argv=[sys.executable, *sys.argv], lane=LANE, memory_max='12G', swap_max=0, threads=1,
            toolchain=TOOLCHAIN, aeneas_model=str(AENEAS_MODEL),
            deadline=os.environ.get('COST_BRIDGE_DEADLINE'),
            renames=[dict(module=name, old=old, new=new) for name, _, old, new in RENAMES],
            controls=[dict(name=c[0], targets=list(c[2])) for c in CONTROLS],
            records=RECORDS, sha256={str(p): digest(p) for p in sources if p.is_file()},
            trust='Lean kernel and toolchain, the Lake package, IO and process handling. The claim '
                  'is about the registered resource algebra only: the counts are source-level visit '
                  'and checked multiply-add counts of the extracted body, and no hardware speed, '
                  'latency, bandwidth, device dispatch cost, cache behaviour or codegen cost '
                  'follows from them.',
        ), indent=2) + '\n')
        shutil.rmtree(SCRATCH, ignore_errors=True)