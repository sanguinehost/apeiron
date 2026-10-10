#!/usr/bin/env python3
"""Shader dot equivalence gate: model scope.
SCOPE: proof/native/ShaderDot.lean, proof/native/shader_dot_gate.py.
REGISTER: arithmetic equivalence between GLSL packed dot and CPU block dot; GLSL translation, GPU execution, codegen excluded.
CORPUS: LaneTrace.lean, LoopTrace.lean, Apeiron.Trichotomy.AnyRing.
REFERENCE: src/spirv/q8_tile.comp, src/engine/chart/portable.rs, proof/corpus/Spirv/Semantics.lean.
GATE: python3 proof/native/shader_dot_gate.py --controls --exact-dot <repo>/target/step-sol-*/aeneas/backends/lean/ExactDotSource.lean; 540s whole invocation.
GRADE: model theorem on signed packed quad and 32-lane block dot; GLSL->SPIR-V translation, GPU execution, codegen excluded.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
OUT = Path('/tmp/opencode/native-shader-dot')
SWARM_TMP = Path(tempfile.gettempdir()) / 'gpu-swarm'

if __name__ == '__main__' and os.environ.get('SHADER_DOT_IN_LANE') != '1':
    now = time.time()
    os.execvp('timeout', [
        'timeout', '--kill-after=5', '540s',
        'flock', '-w', '530', '/tmp/opencode/apeiron-lane.lock',
        'systemd-run', '--user', '--scope', '-q',
        '-p', 'MemoryMax=12G', '-p', 'MemorySwapMax=0',
        'env', 'LEAN_NUM_THREADS=1',
        'SHADER_DOT_IN_LANE=1',
        'SHADER_DOT_START=' + str(now),
        'SHADER_DOT_DEADLINE=' + str(now + 532),
        sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]
    ])

DEADLINE = float(os.environ.get('SHADER_DOT_DEADLINE', time.time() + 532))
STAGES = []; HASHES = {}; ARTIFACTS = []; WORK = None

EXACT_DOT_SHA = '2f955722e75f9a1a101115d05a67e7c3dbe09dd10758a23fe0fdcfc3fded77d2'
ANYRING_SHA = 'decbba9dc965bbf94f83beb982d468e27ddee69c024ecf923740a30217445428'
LANE_TRACE_SHA = '4110c825427fb7c64daeb3981d340756afaccf805ebd17d6bcc5b37186541e2a'

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
        raise RuntimeError('SHADER_DOT_BUDGET_EXHAUSTED')
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
    print('SHADER_DOT_STAGE', label, 'exit=' + str(p.returncode), f'seconds={time.monotonic() - begin:.3f}', str(log), flush=True)
    if interruption is not None:
        raise interruption
    if expected is not None and p.returncode != expected:
        raise RuntimeError(label + '_FAILED ' + str(p.returncode) + '\n' + text[-7000:])
    return p.returncode, text

def main():
    global WORK
    parser = argparse.ArgumentParser()
    parser.add_argument('--controls', action='store_true')
    parser.add_argument(
        '--exact-dot', required=True, metavar='PATH',
        help=('ExactDotSource.lean produced by a step_gate.py run: '
              '<repo>/target/step-sol-*/aeneas/backends/lean/ExactDotSource.lean. The gate takes the '
              'path as an argument instead of naming one historical tempfile.mkdtemp result, and '
              'still requires its content to hash to EXACT_DOT_SHA.'))
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    SWARM_TMP.mkdir(parents=True, exist_ok=True)
    WORK = Path(tempfile.mkdtemp(prefix='shader-dot-', dir=SWARM_TMP))
    print('SHADER_DOT_WORK', WORK, flush=True)

    if not args.controls:
        raise RuntimeError('SHADER_DOT_CONTROLS_REQUIRED')

    shader_dot_src = ROOT / 'proof/native/ShaderDot.lean'
    lane_trace_src = ROOT / 'proof/native/LaneTrace.lean'
    loop_trace_src = ROOT / 'proof/native/LoopTrace.lean'
    exact_dot_src = Path(args.exact_dot).resolve()
    # The extracted body must be the one a step_gate.py run wrote into its own work tree; a
    # hand-placed or unrelated copy cannot stand in for a fresh extraction.
    work_tree = exact_dot_src.parent.parent.parent.parent
    if (exact_dot_src.name != 'ExactDotSource.lean' or exact_dot_src.parent.name != 'lean'
            or exact_dot_src.parent.parent.name != 'backends'
            or exact_dot_src.parent.parent.parent.name != 'aeneas'
            or not work_tree.name.startswith('step-sol-')
            or work_tree.parent != (MAIN / 'target')):
        raise RuntimeError('SHADER_DOT_EXACT_DOT_NOT_A_STEP_WORK_TREE ' + str(exact_dot_src)
                           + ' (want <repo>/target/step-sol-*/aeneas/backends/lean/ExactDotSource.lean)')
    anyring_src = MAIN / 'proof/corpus/.lake/packages/apeiron/proof/Apeiron/Trichotomy/AnyRing.lean'
    pins_file = ROOT / 'proof/patches/PINS'
    seed_cache = MAIN / 'target/native-step-model-cache'

    for p in [shader_dot_src, lane_trace_src, loop_trace_src, exact_dot_src, anyring_src, pins_file]:
        if not p.is_file():
            raise RuntimeError('SHADER_DOT_INPUT_MISSING ' + str(p))
        bind(p)

    # Check sha256 bindings
    if HASHES[str(exact_dot_src)] != EXACT_DOT_SHA:
        raise RuntimeError(f'SHADER_DOT_EXACT_DOT_DRIFT: expected {EXACT_DOT_SHA}, got {HASHES[str(exact_dot_src)]}')
    if HASHES[str(anyring_src)] != ANYRING_SHA:
        raise RuntimeError(f'SHADER_DOT_ANYRING_DRIFT: expected {ANYRING_SHA}, got {HASHES[str(anyring_src)]}')
    if HASHES[str(lane_trace_src)] != LANE_TRACE_SHA:
        raise RuntimeError(f'SHADER_DOT_LANE_TRACE_DRIFT: expected {LANE_TRACE_SHA}, got {HASHES[str(lane_trace_src)]}')

    # Check source text policy on ShaderDot
    sd_text = shader_dot_src.read_text()
    if re.search(r'\b(sorry|admit|axiom|native_decide)\b|set_option\s+(maxHeartbeats|maxRecDepth)', sd_text):
        raise RuntimeError('SHADER_DOT_PROOF_POLICY')

    # Prepare private COW model copy in throwaway work directory
    model = WORK / 'model'
    run(['cp', '-a', '--reflink=always', seed_cache, model], ROOT, 'private-model-cow')
    for p in model.rglob('*'):
        if p.is_symlink() and not p.resolve().is_relative_to(model.resolve()):
            raise RuntimeError('SHADER_DOT_SHARED_CACHE_SYMLINK ' + str(p))

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
    shutil.copy(shader_dot_src, model / 'ShaderDot.lean')
    bind(model / 'ShaderDot.lean')

    # Configure lakefile
    config = model / 'lakefile.lean'
    config.write_text(config.read_text() + '''
lean_lib Apeiron where
  roots := #[`Apeiron.Linter.Registry, `Apeiron.Linter.Citations, `Apeiron.Linter.Notation, `Apeiron.Linter.Citation, `Apeiron.Trichotomy.AnyRing]

lean_lib NativeLaneTrace where
  roots := #[`LoopTrace, `ExactDotSource, `LaneTrace]

lean_lib NativeShaderDot where
  roots := #[`ShaderDot]
''')
    bind(config)

    # Toolchain checks
    run(['lake', 'env', 'lean', '--version'], model, 'lean-version')
    run(['git', 'rev-parse', 'HEAD'], model / '.lake/packages/mathlib', 'mathlib-version')

    # Stage 1: Positive build of Apeiron and NativeShaderDot
    code, text = run(['lake', 'build', 'Apeiron:leanArts', 'NativeShaderDot:leanArts'], model, 'positive-proof')
    if 'SHADER_DOT_KERNEL_OK declarations=' not in text:
        raise RuntimeError('SHADER_DOT_AUDIT_MARKER_MISSING')
    print('ShaderDot positive proof checked successfully under closed axiom cone', flush=True)

    # Stage 2: Negative control on signedness of sign8 (omit - 256, treating byte as unsigned)
    shader_dot_copy = model / 'ShaderDot.lean'
    original_bytes = shader_dot_copy.read_bytes()
    needle = b'if b < 128 then (b : Int) else (b : Int) - 256'
    replacement = b'(b : Int)'
    if original_bytes.count(needle) != 1:
        raise RuntimeError('SHADER_DOT_MUTATION_ANCHOR_MISSING')
    mutated_bytes = original_bytes.replace(needle, replacement, 1)
    shader_dot_copy.write_bytes(mutated_bytes)
    bind(shader_dot_copy)

    red_diagnostic = None
    try:
        code_red, text_red = run(['lake', 'build', 'NativeShaderDot:leanArts'], model, 'mutant-proof', expected=None)
        if code_red != 1 or 'is not definitionally equal to the right-hand side' not in text_red:
            raise RuntimeError('SHADER_DOT_CONTROL_UNATTRIBUTED\n' + text_red[-7000:])
        red_diagnostic = 'packed_dot_eval_concrete rfl rejected unsigned byte mutant (expected -4, got 1020)'
        print('SHADER-DOT control red: ' + red_diagnostic, flush=True)
    finally:
        # Stage 3: Restore exact bytes
        shader_dot_copy.write_bytes(original_bytes)
        if shader_dot_copy.read_bytes() != original_bytes or digest(shader_dot_copy) != digest(shader_dot_src):
            raise RuntimeError('SHADER_DOT_RESTORE_DRIFT')
        bind(shader_dot_copy)

    code_restored, text_restored = run(['lake', 'build', 'NativeShaderDot:leanArts'], model, 'restored-proof')
    if code_restored != 0 or 'SHADER_DOT_KERNEL_OK declarations=' not in text_restored:
        raise RuntimeError('SHADER_DOT_RESTORATION_FAILED')
    print('SHADER-DOT restoration green', flush=True)

    # Post-check: ensure bound source identities did not drift
    exact_sha = digest(exact_dot_src)
    anyring_sha = digest(anyring_src)
    lane_trace_sha = digest(lane_trace_src)
    shader_dot_sha = digest(shader_dot_src)
    if exact_sha != EXACT_DOT_SHA:
        raise RuntimeError('SHADER_DOT_POST_EXACT_DOT_DRIFT')
    if anyring_sha != ANYRING_SHA:
        raise RuntimeError('SHADER_DOT_POST_ANYRING_DRIFT')
    if lane_trace_sha != LANE_TRACE_SHA:
        raise RuntimeError('SHADER_DOT_POST_LANE_TRACE_DRIFT')

    start_time = float(os.environ.get('SHADER_DOT_START', time.time()))
    elapsed = time.time() - start_time
    print(f'ShaderDot source sha256 ExactDotSource: {exact_sha}', flush=True)
    print(f'ShaderDot source sha256 AnyRing: {anyring_sha}', flush=True)
    print(f'ShaderDot source sha256 LaneTrace: {lane_trace_sha}', flush=True)
    print(f'ShaderDot source sha256 ShaderDot: {shader_dot_sha}', flush=True)
    print(f'ShaderDot elapsed: {elapsed:.3f}s', flush=True)
    print('SHADER_DOT_GATE_OK', flush=True)

if __name__ == '__main__':
    status = 1
    try:
        main()
        status = 0
    except BaseException as e:
        print('ShaderDot blocker: ' + str(e), file=sys.stderr, flush=True)
        raise
    finally:
        OUT.mkdir(parents=True, exist_ok=True)
        start_time = float(os.environ.get('SHADER_DOT_START', time.time()))
        elapsed = time.time() - start_time
        receipt_path = OUT / ((WORK.name if WORK else 'shader-dot-no-work') + '-receipt.json')
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
            trust='model-scope signed packed quad and 32-lane block dot equivalence; GLSL->SPIR-V translation, GPU execution, codegen excluded'
        ), indent=2) + '\n')
        # Prune throwaway work directory
        if WORK and WORK.exists():
            shutil.rmtree(WORK, ignore_errors=True)
            print('SHADER_DOT_WORK_PRUNED', flush=True)
