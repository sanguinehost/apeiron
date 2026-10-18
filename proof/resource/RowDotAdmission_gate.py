#!/usr/bin/env python3
"""Check the row-dot replacement admission in a bounded private build.

The admission proposes one change to `src/engine/ops/matvec.rs`: in `row_dot_fixed`'s loop, replace
`match a.checked_mul(b) { … }` by `let prod = a * b; match acc.checked_add(prod) { … }`, keeping the
checked addition and every refusal. This gate

1. refuses to run unless `src/engine/ops/matvec.rs` still contains the production `match a.checked_mul(b)`
   exactly once, so the change is not in production and the admitted baseline is the frozen one;
2. extracts the proposed body with the pinned Charon/Aeneas from a throwaway copy of that source,
   carrying the one-line change and nothing else, and compares the result byte for byte with
   `proof/resource/RowDotProposedSource.lean`, which is what the certificate imports;
3. counts the checked `i128` operations and the indexed row reads in *both* extracted bodies and
   requires the counts the certificate's per-visit constants rest on (1 and 1 checked operations, 2
   and 1, and 2 reads each);
4. builds `proof/resource/RowDotAdmission.lean` in a private Lake package with every headline audited
   under {propext, Classical.choice, Quot.sound}; and
5. seeds two defects in a copy of the certificate — an unsound "the product always fits" claim over
   `I128` operands instead of the `I64` ones, and a per-visit checked-operation count of 3 instead of
   2 — requires both to name and break their own headline, and requires the restored copy green.

The two certificate builds of step 4 and step 5 are independent scratch packages with independent
build trees, so they compile at the same time instead of one after the other; step 5's package is then
reverted to the committed bytes in place and rebuilt, so the restore is still the same package that
went red. No leg inherits another's build product: every leg drops this certificate's own artifacts
first and must show Lake compiling the module, never replaying it. That is what lets the wall clock
fall without any leg's evidence changing.

The claim is about source-level checked-arithmetic counts. No hardware speed follows from it.
`--module` and `--proposed` override the two certificate paths for a pre-integration check; the gate
of record is the invocation without them, which reads the committed files. `--cache-probe` is the
separate control on the build-freshness rule above; it is not part of `--check --controls`.
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
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
GATE = Path(__file__).resolve()
MODULE = ROOT / 'proof/resource/RowDotAdmission.lean'
PROPOSED = ROOT / 'proof/resource/RowDotProposedSource.lean'
PRODUCTION = ROOT / 'src/engine/ops/matvec.rs'
ROW_DOT_SOURCE = ROOT / 'proof/resource/RowDotSource.lean'
ROW_DOT_RESOURCE = ROOT / 'proof/resource/RowDotResource.lean'
COST_ALGEBRA = ROOT / 'proof/resource/CostAlgebra.lean'
COST_BRIDGE = ROOT / 'proof/resource/CostBridge.lean'
AENEAS_MODEL = ROOT / 'target/aeneas-model/lean'
LANE = '/tmp/opencode/apeiron-lane.lock'
TOOLCHAIN = 'leanprover/lean4:v4.31.0'
MARKER = 'ROW_DOT_ADMISSION_KERNEL_OK'
AUDIT = 'ROW_DOT_ADMISSION_AUDIT'
REJECT = 'ROW_DOT_ADMISSION_AXIOM_REJECT'
MODULE_IMPORTS = ('import RowDotProposedSource', 'import RowDotResource', 'import CostBridge')
BUDGET = 540
INNER = 525
# The legs that compile the certificate, as opposed to the legs that extract its Rust body.
CERTIFICATE_LEGS = ('fresh-proof', 'mutant-controls', 'restored',
                    'probe-base', 'probe-inert', 'probe-mutant')

# The identities this admission is bound to (HEAD d5cf5d9). A drift in any of them makes the
# evidence about a different object, so the gate refuses rather than re-binding silently.
FROZEN = {
    PRODUCTION: '34217c1cebb016566b3c485ae6e5323dd54bc8dc8a170fd6a8ec5087b8cd4ec1',
    ROW_DOT_SOURCE: '3dabece20cb5c7e3993c1c42d2d52945a62190e2f3c9e3c46b16be686a8b8b4f',
    ROW_DOT_RESOURCE: '7c561036745c40ebc7db9b2e5bfc298f370fa052dddd43aa37e33eb680fa2f12',
    COST_ALGEBRA: 'c362ec455954e512c2f2055c04e320f26f70529700ea592aeac7eb7a1b422c95',
    COST_BRIDGE: '031b4aab813c5074dfca04936327214bf6262a46e09d83d395e3a7a48c501e17',
}

if __name__ == '__main__' and os.environ.get('ROW_DOT_ADMISSION_IN_LANE') != '1':
    deadline = str(time.time() + (INNER + 5))
    os.execvp('timeout', ['timeout', '--kill-after=5', str(BUDGET) + 's', 'flock', '-w',
        str(INNER + 5), LANE, 'systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=12G',
        '-p', 'MemorySwapMax=0', 'timeout', '--kill-after=5', str(INNER) + 's', 'env',
        'LEAN_NUM_THREADS=1', 'ROW_DOT_ADMISSION_IN_LANE=1',
        'ROW_DOT_ADMISSION_DEADLINE=' + deadline, sys.executable, str(GATE), *sys.argv[1:]])

START = time.monotonic()
RECORDS = []
OUT = Path('/tmp/opencode/rowdot-admission')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path(tempfile.gettempdir()) / 'rowdot-admission'
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='gate-', dir=SCRATCH_ROOT))

# The production loop the change is proposed against, and the one-line replacement.
NEEDLE = """        match a.checked_mul(b) {
            Some(prod) => match acc.checked_add(prod) {
                Some(total) => acc = total,
                None => acc_ok = false,
            },
            None => acc_ok = false,
        }
"""
REPLACEMENT = """        let prod = a * b;
        match acc.checked_add(prod) {
            Some(total) => acc = total,
            None => acc_ok = false,
        }
"""
# The Aeneas model generates each `@[discriminant isize]` instance into the root namespace under the
# declaration's short name, so the two extractions collide on import. The package copy removes
# exactly these five attribute lines; the gate refuses unless that is the whole difference.
DISCRIMINANT = '@[discriminant isize]\n'
PACKAGE_TRANSFORM = ('@[discriminant isize]', 5)

# The single identifier renamed in each dependency copy, and why: two modules registering one
# elaborator name cannot be imported together.
RENAMES = (
    ('CostAlgebra.lean', COST_ALGEBRA, '#resource_audit ', '#cost_algebra_audit '),
    ('RowDotResource.lean', ROW_DOT_RESOURCE, '#resource_audit ', '#row_dot_resource_audit '),
    ('CostBridge.lean', COST_BRIDGE, '#resource_audit ', '#cost_bridge_audit '),
)

# Each control: one single-needle defect seeded in the throwaway certificate copy, and the headline
# that must go red with it. Both are seeded in the same mutant, so one red build carries both.
CONTROLS = (
    ('unsound-product-fits', 'theorem widened_product_fits (iw ix : I64) :',
     'theorem widened_product_fits (iw ix : I128) :',
     ('RowDotAdmission.widened_product_fits',)),
    ('wrong-cost-count',
     'def srcOps (w_row x : Slice I64) (s : I128 × Usize × Bool) : Nat :=\n'
     '  match (srcBodyFn w_row x s).match with\n'
     '  | MatchResult.ok (ControlFlow.cont _) => 2',
     'def srcOps (w_row x : Slice I64) (s : I128 × Usize × Bool) : Nat :=\n'
     '  match (srcBodyFn w_row x s).match with\n'
     '  | MatchResult.ok (ControlFlow.cont _) => 3',
     ('RowDotAdmission.srcOps_of_cont',)),
)

# The per-visit counts the certificate's constants rest on, counted in the extracted body text.
BODY_COUNTS = {
    'srcBodyFn': {'I128.checked_mul': 1, 'I128.checked_add': 1, 'Slice.index_usize': 2},
    'propBodyFn': {'I128.checked_mul': 0, 'I128.checked_add': 1, 'Slice.index_usize': 2},
}

ERROR_KINDS = ('unsolved goals', 'Application type mismatch', 'Type mismatch',
               'Tactic `decide` proved', "'show' tactic failed", 'Tactic `cases` failed',
               'does not synthesize', 'failed to synthesize', 'failed to prove index is valid',
               'Tactic `rw` failed', 'Invalid field', 'Tactic `simp` failed')
DECL_RE = re.compile(r'^(?:@\[\w+\]\s*)?(?:theorem|def|abbrev|structure|inductive|instance)\b', re.M)
LOCATION_RE = re.compile(r'(?:^|\s)(?:error:\s)?RowDotAdmission\.lean:(\d+):(\d+): (.*)$', re.M)
FORBIDDEN_RE = re.compile(r'\b(sorry|axiom|admit)\b|set_option maxHeartbeats 0')
# Raised recursion or heartbeat budgets defeat the reduction guard. The committed extractions are
# exempt because their own generator writes them; this gate's certificate is not.
DEPTH_RE = re.compile(r'maxRecDepth|maxHeartbeats')
BODY_RE = re.compile(r'def matvec\.row_dot_fixed_loop\.body\b.*?\n  := do\n(.*?)\n\n', re.S)

LIB_TEMPLATE = '''pub mod format {{
    pub use super::{{gguf, quant}};
}}
pub mod engine {{
    pub use super::{{dequant, error, f32_convert, matvec, value}};
    pub use super::error::EngineError;
    pub use super::matvec::MatvecError;
}}
#[path = {quant}]
pub mod quant;
#[path = {gguf}]
pub mod gguf;
#[path = {value}]
pub mod value;
#[path = {f32_convert}]
pub mod f32_convert;
#[path = {error}]
pub mod error;
#[path = {dequant}]
pub mod dequant;
#[path = {matvec}]
pub mod matvec;
pub use error::EngineError;

/// Typed forwarding wrapper: it fixes the slice and result types and calls the real
/// `src/engine/ops/matvec.rs` `row_dot_fixed` body. It restates no formula.
pub fn row_dot_forward(w_row: &[i64], x: &[i64]) -> Result<i64, engine::matvec::MatvecError> {{
    engine::matvec::row_dot_fixed(w_row, x)
}}
'''


def digest(path):
    """O(B) bytes; B is one file."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def left():
    """Seconds left in the whole invocation, lane wait included."""
    return min(INNER - (time.monotonic() - START),
               float(os.environ.get('ROW_DOT_ADMISSION_DEADLINE', time.time() + INNER))
               - time.time() - 3)


def run(argv, cwd, label, require=True, certificate=None):
    """Run inside the shared budget and keep the complete output.

    `certificate` is the sha256 of the certificate bytes this leg installed; it is recorded with
    the leg that installed them, which is what ties a leg's log to the source it answered for.
    """
    remaining = left()
    if remaining <= 0:
        raise RuntimeError('ROW_DOT_ADMISSION_BUDGET_EXHAUSTED')
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
    RECORDS.append(dict(label=label, argv=[str(a) for a in argv], cwd=str(cwd), exit=proc.returncode,
                        seconds=round(time.monotonic() - begin, 3), log=str(log),
                        certificate_sha256=certificate))
    if require and proc.returncode:
        raise RuntimeError(f'{label}_REJECT {proc.returncode}\n{text[-3000:]}')
    return proc.returncode, text


def lib_source(matvec):
    """Absolute `#[path]`s to the real modules; only `matvec` varies with the copy."""
    src = ROOT / 'src'
    p = lambda rel: json.dumps(str(src / rel))
    return LIB_TEMPLATE.format(quant=p('format/quant.rs'), gguf=p('format/gguf.rs'),
                               value=p('engine/value.rs'), f32_convert=p('engine/f32_convert.rs'),
                               error=p('engine/error.rs'), dequant=p('engine/dequant.rs'),
                               matvec=json.dumps(str(matvec)))


def extract_proposed(label):
    """Extract the proposed body from a throwaway copy of the real source: the one-line change."""
    original = PRODUCTION.read_text()
    if original.count(NEEDLE) != 1:
        raise RuntimeError('ROW_DOT_ADMISSION_PRODUCTION_TARGET_MISSING ' + str(original.count(NEEDLE)))
    mutant = SCRATCH / 'matvec.rs'
    mutant.write_text(original.replace(NEEDLE, REPLACEMENT))
    project = SCRATCH / 'proposed'
    project.mkdir()
    (project / 'Cargo.toml').write_text('[package]\nname="row_dot_proposed"\nversion="0.0.0"\n'
                                        'edition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project / 'lib.rs').write_text(lib_source(mutant))
    run(['cargo', 'generate-lockfile', '--offline'], project, label + '-lock')
    run(['cargo', 'test', '--locked', '--offline', '--lib', '--target-dir',
         str(SCRATCH / 'rust-target')], project, label + '-rust-tests')
    llbc = project / 'row_dot_proposed.llbc'
    run([str(ROOT / 'target/aeneas/charon'), 'cargo', '--sysroot', 'default', '--preset', 'aeneas',
         '--dest-file', str(llbc), '--start-from', 'crate::row_dot_forward', '--',
         '--manifest-path', str(project / 'Cargo.toml'), '--locked', '--offline', '--lib',
         '--target-dir', str(SCRATCH / 'charon-target')], project, label + '-charon')
    run([str(ROOT / 'target/aeneas/aeneas'), '-backend', 'lean', '-dest', str(project),
         '-warnings-as-errors', '-no-progress-bar', '-all-computable', str(llbc)], project,
        label + '-aeneas')
    generated = project / 'RowDotProposed.lean'
    if not generated.is_file():
        raise RuntimeError('ROW_DOT_ADMISSION_GENERATION_MISSING')
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    run(['python3', str(ROOT / 'proof/strip-modeled-axioms.py'), str(generated)], project,
        label + '-strip')
    generated.write_text(generated.read_text().replace(str(ROOT) + '/', '').replace(str(mutant),
                                                                                  'src/engine/ops/matvec.rs'))
    text = generated.read_text()
    if re.search(r'\b(axiom|sorry|admit)\b', text):
        raise RuntimeError('ROW_DOT_ADMISSION_UNMODELED_SOURCE')
    token, count = PACKAGE_TRANSFORM
    if text.count(token + '\n') != count:
        raise RuntimeError('ROW_DOT_ADMISSION_TRANSFORM_TARGET_MISSING')
    return text, text.replace(token + '\n', ''), digest(mutant)


def body_counts(path):
    """The checked `i128` operations and indexed row reads in one extracted loop body."""
    match = BODY_RE.search(path.read_text())
    if match is None:
        raise RuntimeError('ROW_DOT_ADMISSION_BODY_MISSING ' + str(path))
    body = match.group(1)
    return {token: body.count(token) for token in ('I128.checked_mul', 'I128.checked_add',
                                                    'Slice.index_usize')}


def renamed_copy(name, path, old, new):
    """The dependency copy with one identifier renamed, proved equal to the committed file."""
    original = path.read_text()
    if original.count(old) < 2:
        raise RuntimeError('ROW_DOT_ADMISSION_RENAME_TARGET_MISSING ' + name)
    candidate = original.replace(old, new)
    if candidate.replace(new, old) != original:
        raise RuntimeError('ROW_DOT_ADMISSION_RENAME_NOT_EXACT ' + name)
    return candidate


def preflight(module, proposed):
    """Pin the toolchain, the frozen identities, the module policy and the source precondition."""
    toolchain = AENEAS_MODEL / 'lean-toolchain'
    if not toolchain.is_file() or toolchain.read_text().strip() != TOOLCHAIN:
        raise RuntimeError('ROW_DOT_ADMISSION_TOOLCHAIN_REJECT')
    for path in (ROOT / 'proof/dispatch/lean-toolchain',
                 ROOT / 'proof/dispatch/lake-manifest.json'):
        if not path.is_file():
            raise RuntimeError('ROW_DOT_ADMISSION_LANE_ASSET_MISSING ' + str(path))
    for path, expected in FROZEN.items():
        if not path.is_file():
            raise RuntimeError('ROW_DOT_ADMISSION_FROZEN_MISSING ' + str(path))
        found = digest(path)
        if found != expected:
            raise RuntimeError('ROW_DOT_ADMISSION_FROZEN_DRIFT ' + str(path) + ' ' + found)
    for path in (module, proposed, ROW_DOT_SOURCE, ROW_DOT_RESOURCE, COST_ALGEBRA, COST_BRIDGE):
        if FORBIDDEN_RE.search(path.read_text()):
            raise RuntimeError('ROW_DOT_ADMISSION_MODULE_POLICY_REJECT ' + str(path))
    if DEPTH_RE.search(module.read_text()):
        raise RuntimeError('ROW_DOT_ADMISSION_BUDGET_OPTION_REJECT ' + str(module))
    imports = [line for line in module.read_text().splitlines() if line.startswith('import ')]
    if tuple(imports) != MODULE_IMPORTS:
        raise RuntimeError('ROW_DOT_ADMISSION_IMPORT_CONTRACT_DRIFT ' + repr(imports))
    if len(re.findall(r'^run_cmd Lean\.logInfo "' + MARKER + r'"$', module.read_text(), re.M)) != 1:
        raise RuntimeError('ROW_DOT_ADMISSION_MARKER_MISSING')
    source = PRODUCTION.read_text()
    if source.count(NEEDLE) != 1 or 'let prod = a * b;' in source:
        raise RuntimeError('ROW_DOT_ADMISSION_PRODUCTION_NOT_UNCHANGED')
    for _, needle, _, _ in CONTROLS:
        if module.read_text().count(needle) != 1:
            raise RuntimeError('ROW_DOT_ADMISSION_CONTROL_TARGET_MISSING ' + needle[:60])
    for _, path, old, new in RENAMES:
        renamed_copy(path.name, path, old, new)
    for name, expected in BODY_COUNTS.items():
        source_path = ROW_DOT_SOURCE if name == 'srcBodyFn' else proposed
        found = body_counts(source_path)
        if found != expected:
            raise RuntimeError(f'ROW_DOT_ADMISSION_BODY_COUNT_DRIFT {name} {found}')
    return module.read_text()


def prepare(module_text, package_proposed, name='lean'):
    """Private package; dependency artifacts are read from the pinned lane, never rebuilt.

    `name` keeps concurrent legs in separate packages, each with its own `.lake/build`.
    """
    package = SCRATCH / name
    package.mkdir()
    (package / 'lean-toolchain').write_bytes((AENEAS_MODEL / 'lean-toolchain').read_bytes())
    (package / 'lakefile.lean').write_text(
        'import Lake\nopen Lake DSL\npackage rowDotAdmission where\n'
        '  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\n'
        'require aeneas from ' + json.dumps(str(AENEAS_MODEL)) + '\n'
        '@[default_target] lean_lib RowDotAdmission where\n  roots := #[`CostAlgebra, `RowDotSource,'
        ' `RowDotResource, `RowDotProposedSource, `CostBridge, `RowDotAdmission]\n')
    manifest = json.loads((ROOT / 'proof/dispatch/lake-manifest.json').read_text())
    manifest['name'] = 'rowDotAdmission'
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
    (package / 'RowDotProposedSource.lean').write_text(package_proposed)
    clean_build_artifacts(package)
    (package / 'RowDotAdmission.lean').write_text(module_text)
    return package


def clean_build_artifacts(package):
    """Drop this certificate's products so a mutated copy is certainly recompiled. The renamed
    dependency copies keep their oleans: only the certificate is mutated."""
    root = package / '.lake/build'
    for sub in ('lib/lean', 'ir'):
        directory = root / sub
        if not directory.is_dir():
            continue
        for path in directory.glob('RowDotAdmission.*'):
            if path.is_file():
                path.unlink()


def build(package, text, label, require=True):
    """Install one certificate copy, compile it from scratch in this package, and return the log.

    `clean_build_artifacts` drops this certificate's own olean first, so the leg cannot inherit the
    previous leg's build; the log must then show Lake compiling the module. A leg that reports a
    replay would be answering with another byte sequence's build, and the gate refuses.
    """
    installed = hashlib.sha256(text.encode()).hexdigest()
    (package / 'RowDotAdmission.lean').write_text(text)
    clean_build_artifacts(package)
    code, log = run(['lake', 'build', 'RowDotAdmission'], package, label, require=require,
                    certificate=installed)
    if 'Replayed RowDotAdmission' in log:
        raise RuntimeError(label + '_CACHED_BUILD_SERVED ' + installed + '\n' + log[-2000:])
    return code, log


def build_at_once(jobs):
    """Compile several independent scratch packages at the same time.

    Each job is `(package, text, label, require)` and gets its own Lake package with its own build
    tree, so no two legs share a build product; the only state they share is the pinned dependency
    tree, which they read. Every leg is still a full `lake build` of the real module from its own
    scratch copy, so running two at once changes the wall clock and nothing else. A leg that raises
    is re-raised here, after both legs have stopped.
    """
    results, failures = {}, []

    def leg(job):
        try:
            results[job[2]] = build(*job)
        except BaseException as failure:                      # re-raised below, in the caller
            failures.append(failure)

    workers = [threading.Thread(target=leg, args=(job,)) for job in jobs]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if failures:
        raise failures[0]
    return results


def audited_headlines(module):
    """Every headline the certificate asks to have audited, in source order."""
    return re.findall(r'^#resource_audit ([\w.]+)$', module, re.M)


def declaration_span(module, headline):
    """The line range of one headline's own declaration, its docstring included."""
    name = headline.split('.')[-1]
    match = re.search(r'^(?:theorem|def) ' + re.escape(name) + r'\b', module, re.M)
    if match is None:
        raise RuntimeError('ROW_DOT_ADMISSION_TARGET_HEADLINE_MISSING ' + headline)
    start = module.count('\n', 0, match.start()) + 1
    following = [m.start() for m in DECL_RE.finditer(module) if m.start() > match.start()]
    end = module.count('\n', 0, following[0]) if following else module.count('\n')
    return start, end


def green(text, module, label):
    """A green build: nothing red, one audit per requested headline, one marker."""
    if REJECT in text or 'AXIOM_REJECT' in text or 'forbidden axioms' in text:
        raise RuntimeError(label + '_AUDIT_REJECT\n' + text[-2000:])
    missing = [name for name in audited_headlines(module) if AUDIT + ' ' + name + ':' not in text]
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
    """Each seeded defect must name its own headline and place a diagnostic in its declaration."""
    if code == 0:
        raise RuntimeError(label + '_CONTROL_GREEN\n' + text[-1500:])
    if 'error:' not in text:
        raise RuntimeError(label + '_CONTROL_NO_ERROR\n' + text[-1500:])
    locations = diagnostic_lines(text)
    if not locations:
        raise RuntimeError(label + '_CONTROL_UNATTRIBUTED\n' + text[-2000:])
    unnamed = [name for name in targets if REJECT + ' ' + name + ':' not in text]
    if unnamed:
        raise RuntimeError(label + '_CONTROL_HEADLINE_NOT_NAMED ' + repr(unnamed) + '\n' + text[-2000:])
    out_of_span = [name for name in targets
                   if not any(declaration_span(module, name)[0] <= line
                              <= declaration_span(module, name)[1] for line in locations)]
    if out_of_span:
        raise RuntimeError(label + '_CONTROL_DIAGNOSTIC_OUTSIDE_HEADLINE ' + repr(out_of_span)
                           + ' locations=' + repr(locations) + '\n' + text[-2000:])
    if not any(kind in text for kind in ERROR_KINDS):
        raise RuntimeError(label + '_CONTROL_NO_DIAGNOSTIC\n' + text[-2000:])
    return locations


def mutant_certificate(module):
    """The certificate with every seeded defect applied at once, one red build carrying both."""
    mutant = module
    for name, needle, replacement, targets in CONTROLS:
        mutant = mutant.replace(needle, replacement)
    if mutant == module:
        raise RuntimeError('ROW_DOT_ADMISSION_CONTROL_MUTATION_EMPTY')
    return mutant


def prove(text, module, label):
    """A green leg: the audits, the marker, and the three renamed dependency markers."""
    green(text, module, label)
    for marker in ('COST_ALGEBRA_KERNEL_OK', 'ROW_DOT_KERNEL_OK', 'COST_BRIDGE_KERNEL_OK'):
        if marker not in text:
            raise RuntimeError(label + '_RENAMED_DEPENDENCY_INCOMPLETE ' + marker + '\n'
                               + text[-2000:])


def cache_probe(module, package_proposed):
    """Control on the build-freshness rule `build` enforces: a package holding a green build of
    other bytes must not answer for a byte sequence it has not compiled.

    Three legs in one scratch package: the committed certificate green, then a third byte sequence
    — the same certificate behind an inert comment — which must be recompiled and stay green, then
    the seeded mutant bytes, which must be recompiled and go red. A cache that ignored the bytes
    would replay leg one for legs two and three; `build` refuses a replayed module, so this control
    ends either in a rebuild or in a refusal.
    """
    package = prepare(module, package_proposed, 'lean-probe')
    targets = [target for _, _, _, group in CONTROLS for target in group]
    code, text = build(package, module, 'probe-base')
    prove(text, module, 'PROBE_BASE')
    inert = module + '\n-- cache probe: an inert third byte sequence, no declaration attached\n'
    if inert == module or not inert.startswith(module):
        raise RuntimeError('ROW_DOT_ADMISSION_PROBE_SEQUENCE_NOT_DISTINCT')
    code, text = build(package, inert, 'probe-inert')
    prove(text, module, 'PROBE_INERT')
    print('ROW_DOT_ADMISSION_CACHE_PROBE_INERT_REBUILT base=%s inert=%s'
          % (hashlib.sha256(module.encode()).hexdigest()[:16],
             hashlib.sha256(inert.encode()).hexdigest()[:16]))
    code, text = build(package, mutant_certificate(module), 'probe-mutant', require=False)
    locations = attributable(code, text, module, targets, 'ROW_DOT_ADMISSION_PROBE_MUTANT')
    print('ROW_DOT_ADMISSION_CACHE_PROBE_MUTANT_RED defects=%s targets=%s lines=%s'
          % (','.join(name for name, _, _, _ in CONTROLS), ','.join(targets), locations))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--controls', action='store_true')
    parser.add_argument('--cache-probe', action='store_true',
                        help='control on the build-freshness rule; not part of --check --controls')
    parser.add_argument('--module', default=str(MODULE),
                        help='certificate path; the gate of record uses the committed file')
    parser.add_argument('--proposed', default=str(PROPOSED),
                        help='proposed extraction path; the gate of record uses the committed file')
    args = parser.parse_args()
    module_path, proposed_path = Path(args.module), Path(args.proposed)
    module = preflight(module_path, proposed_path)
    run(['lake', '--version'], ROOT, 'lake-version')
    generated, package_proposed, mutant_digest = extract_proposed('fresh')
    if proposed_path.read_text() != package_proposed:
        raise RuntimeError('ROW_DOT_ADMISSION_EXTRACTION_DRIFT')
    print('ROW_DOT_ADMISSION_EXTRACTION_OK mutant_sha256=%s committed_sha256=%s'
          % (mutant_digest, digest(proposed_path)))
    headlines = audited_headlines(module)
    package = prepare(module, package_proposed)
    if args.cache_probe:
        cache_probe(module, package_proposed)
    elif args.controls:
        # Two independent packages, compiled together; the mutant's package is then reverted in
        # place, so the restore build is the same package that went red.
        control_package = prepare(module, package_proposed, 'lean-control')
        results = build_at_once(((package, module, 'fresh-proof', True),
                                 (control_package, mutant_certificate(module),
                                  'mutant-controls', False)))
        prove(results['fresh-proof'][1], module, 'FRESH')
        print('ROW_DOT_ADMISSION_PROOF_OK check=%s headlines=%d extracts=%s'
              % (args.check, len(headlines), 'RowDotProposedSource'))
        code, text = results['mutant-controls']
        targets = [target for _, _, _, group in CONTROLS for target in group]
        locations = attributable(code, text, module, targets, 'ROW_DOT_ADMISSION_CONTROLS')
        print('ROW_DOT_ADMISSION_CONTROL_RED defects=%s targets=%s lines=%s'
              % (','.join(name for name, _, _, _ in CONTROLS), ','.join(targets), locations))
        code, text = build(control_package, module, 'restored')
        prove(text, module, 'RESTORED')
        print('ROW_DOT_ADMISSION_CONTROL_RESTORED defects=%d'
              % len(CONTROLS))
    else:
        code, text = build(package, module, 'fresh-proof')
        prove(text, module, 'FRESH')
        print('ROW_DOT_ADMISSION_PROOF_OK check=%s headlines=%d extracts=%s'
              % (args.check, len(headlines), 'RowDotProposedSource'))
    print('ROW_DOT_ADMISSION_LEG_SECONDS %s'
          % ' '.join('%s=%.1f' % (record['label'], record['seconds']) for record in RECORDS
                     if record['label'] in CERTIFICATE_LEGS))
    print('ROW_DOT_ADMISSION_COST_CERTIFICATE_OK')
    print('generated_bytes=%d committed_bytes=%d' % (len(generated.encode()), len(package_proposed.encode())))


if __name__ == '__main__':
    status = 1
    try:
        main()
        status = 0
    finally:
        sources = [MODULE, PROPOSED, GATE, PRODUCTION, ROW_DOT_SOURCE, ROW_DOT_RESOURCE,
                   COST_ALGEBRA, COST_BRIDGE, AENEAS_MODEL / 'lean-toolchain',
                   ROOT / 'proof/dispatch/lake-manifest.json']
        (OUT / (SCRATCH.name + '-receipt.json')).write_text(json.dumps(dict(
            exit=status, seconds=round(time.monotonic() - START, 3), budget=BUDGET,
            argv=[sys.executable, *sys.argv], lane=LANE, memory_max='12G', swap_max=0, threads=1,
            toolchain=TOOLCHAIN, aeneas_model=str(AENEAS_MODEL),
            deadline=os.environ.get('ROW_DOT_ADMISSION_DEADLINE'),
            frozen={str(k): v for k, v in FROZEN.items()},
            renames=[dict(module=name, old=old, new=new) for name, _, old, new in RENAMES],
            package_transform=dict(token=PACKAGE_TRANSFORM[0], count=PACKAGE_TRANSFORM[1]),
            body_counts={k: v for k, v in BODY_COUNTS.items()},
            controls=[dict(name=c[0], targets=list(c[3])) for c in CONTROLS],
            records=RECORDS, sha256={str(p): digest(p) for p in sources if p.is_file()},
            trust='Lean kernel and toolchain, the Lake package, Charon/Aeneas and IO/process handling. '
                  'The claim is about source-level checked-arithmetic counts of the two extracted '
                  'bodies: no hardware speed, machine code, instruction count, latency, bandwidth, '
                  'cache behaviour or codegen cost follows from it. `src/engine/ops/matvec.rs` is '
                  'unchanged; the proposed body is not in production.',
        ), indent=2) + '\n')
        shutil.rmtree(SCRATCH, ignore_errors=True)