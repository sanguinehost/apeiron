#!/usr/bin/env python3
"""Actual extracted inner-loop lane trace: model scope.
SCOPE: proof/native/LaneTrace.lean, proof/native/lane_trace_gate.py.
REGISTER: model lane count and ordered integer sum for extracted inner loop; cost excluded.
CORPUS: LoopTrace.lean, ExactDotSource.lean, Apeiron.Trichotomy.AnyRing.
REFERENCE: target/step-sol-c5hbky9_/aeneas/backends/lean/ExactDotSource.lean,
           Apeiron/Trichotomy/AnyRing.lean, Aeneas/Std/SliceIter.lean.
GATE: python3 proof/native/lane_trace_gate.py --controls; 540s whole invocation.
GRADE: model theorem on extracted body; Rust reachability, GPU, cost excluded.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
OUT = Path('/tmp/opencode/native-lane-trace')
SWARM_TMP = Path(tempfile.gettempdir()) / 'gpu-swarm'

if __name__ == '__main__' and os.environ.get('LANE_TRACE_IN_LANE') != '1':
    now = time.time()
    os.execvp('timeout', [
        'timeout', '--kill-after=5', '540s',
        'flock', '-w', '530', '/tmp/opencode/apeiron-lane.lock',
        'systemd-run', '--user', '--scope', '-q',
        '-p', 'MemoryMax=12G', '-p', 'MemorySwapMax=0',
        'env', 'LEAN_NUM_THREADS=1',
        'LANE_TRACE_IN_LANE=1',
        'LANE_TRACE_START=' + str(now),
        'LANE_TRACE_DEADLINE=' + str(now + 532),
        sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]
    ])

DEADLINE = float(os.environ.get('LANE_TRACE_DEADLINE', time.time() + 532))
STAGES = []; HASHES = {}; ARTIFACTS = []; WORK = None

EXACT_DOT_SHA = '2f955722e75f9a1a101115d05a67e7c3dbe09dd10758a23fe0fdcfc3fded77d2'
ANYRING_SHA = 'decbba9dc965bbf94f83beb982d468e27ddee69c024ecf923740a30217445428'

def digest(path):
    """O(B) input bytes streaming digest."""
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def bind(path):
    value = digest(path)
    HASHES[str(path)] = value
    ARTIFACTS.append(dict(path=str(path), sha256=value, stage=STAGES[-1]['stage'] if STAGES else 'input'))
    return value

def run(argv, cwd, label, expected=0):
    left = DEADLINE - time.time() - 3
    if left <= 0:
        raise RuntimeError('LANE_TRACE_BUDGET_EXHAUSTED')
    log = OUT / ((WORK.name if WORK else 'preflight') + '-' + label + '.log')
    begin = time.monotonic()
    argv = list(map(str, argv))
    with log.open('wb') as f:
        p = subprocess.Popen(
            argv, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
            env={**os.environ, 'LEAN_NUM_THREADS': '1'}, start_new_session=True
        )
        interruption = None
        try:
            p.wait(timeout=left)
        except BaseException as e:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            interruption = e
    text = log.read_text(errors='replace')
    STAGES.append(dict(stage=label, exit=p.returncode, seconds=time.monotonic() - begin, argv=argv, cwd=str(cwd), log=str(log)))
    print('LANE_TRACE_STAGE', label, 'exit=' + str(p.returncode), f'seconds={time.monotonic() - begin:.3f}', str(log), flush=True)
    if interruption is not None:
        raise interruption
    if expected is not None and p.returncode != expected:
        raise RuntimeError(label + '_FAILED ' + str(p.returncode) + '\n' + text[-7000:])
    return p.returncode, text

def main():
    global WORK
    parser = argparse.ArgumentParser()
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    SWARM_TMP.mkdir(parents=True, exist_ok=True)
    WORK = Path(tempfile.mkdtemp(prefix='lane-trace-', dir=SWARM_TMP))
    print('LANE_TRACE_WORK', WORK, flush=True)

    if not args.controls:
        raise RuntimeError('LANE_TRACE_CONTROLS_REQUIRED')

    lane_trace_src = ROOT / 'proof/native/LaneTrace.lean'
    loop_trace_src = ROOT / 'proof/native/LoopTrace.lean'
    exact_dot_src = MAIN / 'target/step-sol-c5hbky9_/aeneas/backends/lean/ExactDotSource.lean'
    anyring_src = MAIN / 'proof/corpus/.lake/packages/apeiron/proof/Apeiron/Trichotomy/AnyRing.lean'
    pins_file = ROOT / 'proof/patches/PINS'
    seed_cache = MAIN / 'target/native-step-model-cache'

    for p in [lane_trace_src, loop_trace_src, exact_dot_src, anyring_src, pins_file]:
        if not p.is_file():
            raise RuntimeError('LANE_TRACE_INPUT_MISSING ' + str(p))
        bind(p)

    # Check sha256 bindings
    if HASHES[str(exact_dot_src)] != EXACT_DOT_SHA:
        raise RuntimeError(f'LANE_TRACE_EXACT_DOT_DRIFT: expected {EXACT_DOT_SHA}, got {HASHES[str(exact_dot_src)]}')
    if HASHES[str(anyring_src)] != ANYRING_SHA:
        raise RuntimeError(f'LANE_TRACE_ANYRING_DRIFT: expected {ANYRING_SHA}, got {HASHES[str(anyring_src)]}')

    # Check source text policy on LaneTrace
    lt_text = lane_trace_src.read_text()
    if re.search(r'\b(sorry|admit|axiom|native_decide)\b|set_option\s+(maxHeartbeats|maxRecDepth)', lt_text):
        raise RuntimeError('LANE_TRACE_PROOF_POLICY')

    # Prepare private COW model copy in throwaway work directory
    model = WORK / 'model'
    run(['cp', '-a', '--reflink=always', seed_cache, model], ROOT, 'private-model-cow')
    for p in model.rglob('*'):
        if p.is_symlink() and not p.resolve().is_relative_to(model.resolve()):
            raise RuntimeError('LANE_TRACE_SHARED_CACHE_SYMLINK ' + str(p))

    # Copy Apeiron package into model
    apeiron_dir = MAIN / 'proof/corpus/.lake/packages/apeiron/proof/Apeiron'
    run(['cp', '-a', apeiron_dir, model / 'Apeiron'], ROOT, 'copy-apeiron')

    # Copy sources into model
    shutil.copy(exact_dot_src, model / 'ExactDotSource.lean')
    bind(model / 'ExactDotSource.lean')
    shutil.copy(loop_trace_src, model / 'LoopTrace.lean')
    bind(model / 'LoopTrace.lean')
    shutil.copy(lane_trace_src, model / 'LaneTrace.lean')
    bind(model / 'LaneTrace.lean')

    # Configure lakefile
    config = model / 'lakefile.lean'
    config.write_text(config.read_text() + '''
lean_lib Apeiron where
  roots := #[`Apeiron.Linter.Registry, `Apeiron.Linter.Citations, `Apeiron.Linter.Notation, `Apeiron.Linter.Citation, `Apeiron.Trichotomy.AnyRing]

lean_lib NativeLaneTrace where
  roots := #[`LoopTrace, `ExactDotSource, `LaneTrace]
''')
    bind(config)

    # Toolchain checks
    run(['lake', 'env', 'lean', '--version'], model, 'lean-version')
    run(['git', 'rev-parse', 'HEAD'], model / '.lake/packages/mathlib', 'mathlib-version')

    # Stage 1: Positive build of Apeiron and NativeLaneTrace
    code, text = run(['lake', 'build', 'Apeiron:leanArts', 'NativeLaneTrace:leanArts'], model, 'positive-proof')
    if 'LANE_TRACE_KERNEL_OK declarations=' not in text:
        raise RuntimeError('LANE_TRACE_AUDIT_MARKER_MISSING')
    print('LaneTrace positive proof checked successfully under closed axiom cone', flush=True)

    # Stage 2: Negative control on exact arithmetic node (+= to -=)
    exact_dot_copy = model / 'ExactDotSource.lean'
    original_bytes = exact_dot_copy.read_bytes()
    needle = b'let dot1 \xe2\x86\x90 dot + i6' # "let dot1 ← dot + i6"
    replacement = b'let dot1 \xe2\x86\x90 dot - i6' # "let dot1 ← dot - i6"
    if original_bytes.count(needle) != 1:
        raise RuntimeError('LANE_TRACE_MUTATION_ANCHOR_MISSING')
    mutated_bytes = original_bytes.replace(needle, replacement, 1)
    exact_dot_copy.write_bytes(mutated_bytes)
    bind(exact_dot_copy)

    red_diagnostic = None
    try:
        code_red, text_red = run(['lake', 'build', 'NativeLaneTrace:leanArts'], model, 'mutant-proof', expected=None)
        if code_red != 1 or 'LANE_TRACE_CLOSED_CONE_REJECT' not in text_red or 'forbidden dependencies [sorryAx]' not in text_red:
            raise RuntimeError('LANE_TRACE_CONTROL_UNATTRIBUTED\n' + text_red[-7000:])
        red_diagnostic = 'LANE_TRACE_CLOSED_CONE_REJECT with forbidden dependencies [sorryAx] on mutant dot - i6'
        print('LaneTrace negative control red: ' + red_diagnostic, flush=True)
    finally:
        # Stage 3: Restore exact bytes
        exact_dot_copy.write_bytes(original_bytes)
        if exact_dot_copy.read_bytes() != original_bytes or digest(exact_dot_copy) != EXACT_DOT_SHA:
            raise RuntimeError('LANE_TRACE_RESTORE_DRIFT')
        bind(exact_dot_copy)

    code_restored, text_restored = run(['lake', 'build', 'NativeLaneTrace:leanArts'], model, 'restored-proof')
    if code_restored != 0 or 'LANE_TRACE_KERNEL_OK declarations=' not in text_restored:
        raise RuntimeError('LANE_TRACE_RESTORATION_FAILED')
    print('LaneTrace restoration green', flush=True)

    # Post-check: ensure bound source identities did not drift
    exact_sha = digest(exact_dot_src)
    anyring_sha = digest(anyring_src)
    lane_trace_sha = digest(lane_trace_src)
    if exact_sha != EXACT_DOT_SHA:
        raise RuntimeError('LANE_TRACE_POST_EXACT_DOT_DRIFT')
    if anyring_sha != ANYRING_SHA:
        raise RuntimeError('LANE_TRACE_POST_ANYRING_DRIFT')

    start_time = float(os.environ.get('LANE_TRACE_START', time.time()))
    elapsed = time.time() - start_time
    print(f'LaneTrace source sha256 ExactDotSource: {exact_sha}', flush=True)
    print(f'LaneTrace source sha256 AnyRing: {anyring_sha}', flush=True)
    print(f'LaneTrace source sha256 LaneTrace: {lane_trace_sha}', flush=True)
    print(f'LaneTrace elapsed: {elapsed:.3f}s', flush=True)

    print('LANE_TRACE_GATE_OK', flush=True)

if __name__ == '__main__':
    status = 1
    try:
        main()
        status = 0
    except BaseException as e:
        print('LaneTrace blocker: ' + str(e), file=sys.stderr, flush=True)
        raise
    finally:
        OUT.mkdir(parents=True, exist_ok=True)
        start_time = float(os.environ.get('LANE_TRACE_START', time.time()))
        elapsed = time.time() - start_time
        receipt_path = OUT / ((WORK.name if WORK else 'lane-trace-no-work') + '-receipt.json')
        receipt_path.write_text(json.dumps(dict(
            exit=status,
            wall_seconds=elapsed,
            budget=540,
            deadline=DEADLINE,
            argv=sys.argv,
            stages=STAGES,
            sha256=HASHES,
            artifacts=ARTIFACTS,
            work=str(WORK),
            memory='12G',
            swap=0,
            threads=1,
            trust='model-scope finite lane trace for extracted exact_dot inner loop; divergence/overflow excluded; whole-function cost, GPU, codegen excluded'
        ), indent=2) + '\n')
        # Prune throwaway work directory to prevent storage exhaustion
        if WORK and WORK.exists():
            shutil.rmtree(WORK, ignore_errors=True)
