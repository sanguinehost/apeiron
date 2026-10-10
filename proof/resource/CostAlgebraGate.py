#!/usr/bin/env python3
"""Check the composition/dominance cost algebra in a bounded private Lean package.

The module is copied into a throwaway Lake package that resolves the pinned toolchain
from target/aeneas-model/lean; only the copy is ever mutated. O(B) module bytes are read
per build; external compilation is timed, not assigned constant cost. Each of the three
seeded composition defects must turn the build red with an attributable diagnostic, and
the restored copy must build green.
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
MODULE = ROOT / 'proof/resource/CostAlgebra.lean'
GATE = Path(__file__).resolve()
CONTRACT = ROOT / 'proof/DispatchAdmission.lean'
AENEAS_MODEL = ROOT / 'target/aeneas-model/lean'
# The certificate shape this module's registered `Nat` cost is stated against.
CERTIFICATE_FIELD = 'improves : ∀ x, reachable x → admissible x → afterCost x < beforeCost x'
TOOLCHAIN = 'leanprover/lean4:v4.31.0'
MARKER = 'COST_ALGEBRA_KERNEL_OK'
AUDIT = 'COST_ALGEBRA_AUDIT'
REJECT = 'COST_ALGEBRA_AXIOM_REJECT'

if __name__ == '__main__' and os.environ.get('COST_ALGEBRA_IN_LANE') != '1':
    deadline = str(time.time() + 535)
    os.execvp('timeout', ['timeout', '--kill-after=5', '540s', 'flock', '-w', '530',
        '/tmp/opencode/apeiron-lane.lock', 'systemd-run', '--user', '--scope', '-q',
        '-p', 'MemoryMax=12G', '-p', 'MemorySwapMax=0', 'timeout', '--kill-after=5', '525s',
        'env', 'LEAN_NUM_THREADS=1', 'COST_ALGEBRA_IN_LANE=1', 'COST_ALGEBRA_DEADLINE=' + deadline,
        sys.executable, str(GATE), *sys.argv[1:]])

START = time.monotonic()
RECORDS = []
OUT = Path('/tmp/opencode/cost-algebra')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path(tempfile.gettempdir()) / 'cost-algebra'
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='gate-', dir=SCRATCH_ROOT))

# Each control: a single-needle text defect seeded in the throwaway module copy, the
# headlines that must go red with it, and the defect's name for the receipt.
CONTROLS = (
    ('duplicated-effect', 'planCost_shared_add',
     'def planCost (p : Plan) : Cost := (p.layers.map layerCost).foldr (· + ·) zero + p.shared',
     'def planCost (p : Plan) : Cost := (p.layers.map layerCost).foldr (· + ·) zero + p.shared + p.shared',
     ('planCost_shared_add', 'resident_cost')),
    ('omitted-packing', 'costAdd',
     '  \u27e8a.ops + b.ops, a.pack + b.pack, a.alloc + b.alloc, a.xfer + b.xfer,\n'
     '    a.dispatch + b.dispatch, a.peak + b.peak, a.path + b.path\u27e9',
     '  \u27e8a.ops + b.ops, 0, a.alloc + b.alloc, a.xfer + b.xfer,\n'
     '    a.dispatch + b.dispatch, a.peak + b.peak, a.path + b.path\u27e9',
     ('packing_reduction', 'perProjection_cost', 'resident_cost')),
    ('non-strict-improvement', 'strictlyCheaper',
     'def strictlyCheaper (a b : Cost) : Prop := a ≤ b ∧ a ≠ b',
     'def strictlyCheaper (a b : Cost) : Prop := a ≤ b',
     ('tie_not_improvement', 'resident_dominates')),
)
ERROR_KINDS = ('unsolved goals', 'Type mismatch', 'Application type mismatch',
               'Tactic `decide` proved', 'Function expected', "tactic failed")


def digest(path):
    """O(B) bytes; B is one module copy."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def left():
    """Seconds left in the whole invocation, lane wait included."""
    return min(525 - (time.monotonic() - START),
               float(os.environ.get('COST_ALGEBRA_DEADLINE', time.time() + 525)) - time.time() - 3)


def run(argv, cwd, label, require=True):
    """Run inside the shared budget and keep the complete output."""
    remaining = left()
    if remaining <= 0:
        raise RuntimeError('COST_ALGEBRA_BUDGET_EXHAUSTED')
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


def preflight():
    """Pin the toolchain, the certificate contract and the module policy before any build."""
    toolchain = AENEAS_MODEL / 'lean-toolchain'
    if not toolchain.is_file() or toolchain.read_text().strip() != TOOLCHAIN:
        raise RuntimeError('COST_ALGEBRA_TOOLCHAIN_REJECT')
    if not (ROOT / 'proof/dispatch/lean-toolchain').is_file():
        raise RuntimeError('COST_ALGEBRA_LANE_MANIFEST_MISSING')
    contract = [line.strip() for line in CONTRACT.read_text().splitlines()]
    if CERTIFICATE_FIELD not in contract:
        raise RuntimeError('COST_ALGEBRA_CERTIFICATE_CONTRACT_DRIFT')
    module = MODULE.read_text()
    if re.search(r'\b(sorry|axiom|admit)\b|set_option maxHeartbeats 0', module):
        raise RuntimeError('COST_ALGEBRA_MODULE_POLICY_REJECT')
    if module.count('run_cmd Lean.logInfo "' + MARKER + '"') != 1:
        raise RuntimeError('COST_ALGEBRA_MARKER_MISSING')
    return module


def prepare():
    """Private package; dependency artifacts are read from the pinned lane, never rebuilt."""
    package = SCRATCH / 'lean'
    package.mkdir()
    (package / 'lean-toolchain').write_bytes((AENEAS_MODEL / 'lean-toolchain').read_bytes())
    (package / 'lakefile.lean').write_text(
        'import Lake\nopen Lake DSL\npackage costAlgebra where\n'
        '  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\n'
        'require aeneas from ' + json.dumps(str(AENEAS_MODEL)) + '\n'
        '@[default_target] lean_lib CostAlgebra where\n  roots := #[`CostAlgebra]\n')
    manifest = json.loads((ROOT / 'proof/dispatch/lake-manifest.json').read_text())
    manifest['name'] = 'costAlgebra'
    for package_entry in manifest['packages']:
        if package_entry['name'] == 'aeneas':
            package_entry['dir'] = str(AENEAS_MODEL)
    (package / 'lake-manifest.json').write_text(json.dumps(manifest))
    packages = package / '.lake/packages'
    packages.mkdir(parents=True)
    for dependency in (ROOT / 'proof/dispatch/.lake/packages').iterdir():
        link = packages / dependency.name
        if not link.exists():
            link.symlink_to(dependency.resolve(), target_is_directory=True)
    return package


def clean_build_artifacts(package):
    """Drop this module's products so a mutated copy is certainly recompiled."""
    root = package / '.lake/build'
    for sub in ('lib/lean', 'ir'):
        directory = root / sub
        if not directory.is_dir():
            continue
        for path in directory.glob('CostAlgebra.*'):
            if path.is_file():
                path.unlink()


def build(package, text, label, require=True):
    """Install one module copy, build it, and return the complete log."""
    (package / 'CostAlgebra.lean').write_text(text)
    clean_build_artifacts(package)
    return run(['lake', 'build', 'CostAlgebra'], package, label, require=require)


def audited_headlines(module):
    """Every headline the module asks to have audited, in source order."""
    return re.findall(r'^#resource_audit ([\w.]+)$', module, re.M)


def green(text, module, label):
    """A green build: no rejected audit, one audit per requested headline, one marker."""
    if REJECT in text:
        raise RuntimeError(label + '_AUDIT_REJECT\n' + text[-2000:])
    missing = [name for name in audited_headlines(module) if AUDIT + ' ' + name + ':' not in text]
    if missing:
        raise RuntimeError(label + '_AUDIT_MISSING ' + repr(missing) + '\n' + text[-2000:])
    if text.count(MARKER) != 1:
        raise RuntimeError(label + '_MARKER_MISSING\n' + text[-2000:])


def attributable(code, text, headlines, label):
    """The seeded defect must name the headline it broke, not fail elsewhere."""
    if code == 0:
        raise RuntimeError(label + '_CONTROL_GREEN ' + text[-1500:])
    if 'CostAlgebra.lean:' not in text:
        raise RuntimeError(label + '_CONTROL_UNATTRIBUTED\n' + text[-2000:])
    named = [name for name in headlines if REJECT + ' CostAlgebra.' + name + ':' in text]
    if not named:
        raise RuntimeError(label + '_CONTROL_HEADLINE_NOT_NAMED ' + repr(headlines) + '\n' + text[-2000:])
    if not any(kind in text for kind in ERROR_KINDS):
        raise RuntimeError(label + '_CONTROL_NO_DIAGNOSTIC\n' + text[-2000:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    module = preflight()
    run(['lake', '--version'], ROOT, 'lake-version')
    package = prepare()
    code, text = build(package, module, 'fresh-proof')
    green(text, module, 'FRESH')
    headlines = len(audited_headlines(module))
    print('COST_ALGEBRA_ALGEBRA_OK check=%s headlines=%d' % (args.check, headlines))
    if args.controls:
        for name, needle_name, needle, replacement, targets in CONTROLS:
            if module.count(needle) != 1:
                raise RuntimeError('COST_ALGEBRA_CONTROL_TARGET_MISSING ' + name)
            code, text = build(package, module.replace(needle, replacement),
                               'mutant-' + name, require=False)
            attributable(code, text, targets, 'COST_ALGEBRA_' + name.upper())
            print('COST_ALGEBRA_CONTROL_RED %s targets=%s' % (name, ','.join(targets)))
            code, text = build(package, module, 'restored-' + name)
            green(text, module, 'RESTORED_' + name.upper())
            print('COST_ALGEBRA_CONTROL_RESTORED %s' % name)
    print('COST_ALGEBRA_COMPOSITION_OK')


if __name__ == '__main__':
    status = 1
    try:
        main()
        status = 0
    finally:
        sources = [MODULE, GATE, CONTRACT, AENEAS_MODEL / 'lean-toolchain',
                   ROOT / 'proof/dispatch/lake-manifest.json']
        (OUT / (SCRATCH.name + '-receipt.json')).write_text(json.dumps(dict(
            exit=status, seconds=round(time.monotonic() - START, 3), budget=540,
            argv=[sys.executable, *sys.argv],
            lane='/tmp/opencode/apeiron-lane.lock', memory_max='12G', swap_max=0, threads=1,
            toolchain=TOOLCHAIN, aeneas_model=str(AENEAS_MODEL),
            deadline=os.environ.get('COST_ALGEBRA_DEADLINE'),
            controls=[dict(name=c[0], target=c[1], red_headlines=list(c[4])) for c in CONTROLS],
            records=RECORDS, sha256={str(p): digest(p) for p in sources if p.is_file()},
            trust='Lean kernel and toolchain, the Lake package, IO and process handling. The claim '
                  'is about the resource algebra only: no hardware speed, no compiler or codegen '
                  'cost, no latency, bandwidth or device dispatch measurement.',
        ), indent=2) + '\n')
        shutil.rmtree(SCRATCH, ignore_errors=True)
