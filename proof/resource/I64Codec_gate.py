#!/usr/bin/env python3
"""Extract and check the real `i64` word-codec encode/decode bodies in a bounded private build.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
The temporary manifest wires the real `src/{scalar,cell,coord,table}.rs` by `#[path]` -- the same
real closure `proof/extract/lib.rs` declares for the two extraction roots -- and adds only the
typed forwarding wrappers `proof/extract/lib.rs` already carries for
`word_i64_codec_encode` / `word_i64_codec_decode`. No substitute codec is extracted.

  proof/resource/I64Codec_gate.py --prepare   regenerate proof/resource/I64CodecSource.lean
  proof/resource/I64Codec_gate.py             check extraction freshness, then build the proof
  proof/resource/I64Codec_gate.py --controls  also run the real-source mutation/restore control

The lane is /tmp/opencode/apeiron-lane-3.lock, 540 s wall clock for the whole invocation,
LEAN_NUM_THREADS=1, systemd-run MemoryMax=12G, MemorySwapMax=0. The work measure in
I64CodecResource.lean counts `Result` steps and 64/128-bit word operations under named premises;
it is not a cycle, wall-clock or GPU measurement.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
LANE = '/tmp/opencode/apeiron-lane-3.lock'
if __name__ == '__main__' and os.environ.get('I64_CODEC_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','I64_CODEC_IN_LANE=1',
        'I64_CODEC_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (failure register: /tmp tmpfs headroom); only named receipts
# and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/i64codec-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path('/home/socol/oc-tmp/i64codec-resource')
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('I64_CODEC_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('I64_CODEC_BUDGET_EXHAUSTED')
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

# The real source closure the wrapper compiles. `src/table.rs` pulls `src/table/keyrep.rs` by
# `#[path]`, which is the module under test; `src/table.rs` itself reads `crate::cell::Sigma` and
# `crate::coord::{address,locator}`, so the real `src/{scalar,cell,coord,table}.rs` are declared
# at the temporary crate root exactly as `proof/extract/lib.rs` declares them. The mutant control
# copies the whole `src/table/` tree and `src/table.rs` into scratch and mutates the copied
# `keyrep.rs`, never the working tree.
LIB_TEMPLATE = '''#[path = {scalar}]
pub mod scalar;
#[path = {cell}]
pub mod cell;
#[path = {coord}]
pub mod coord;
#[path = {table}]
pub mod table;
/// The real `src/table.rs` test modules read `crate::Sigma`; the alias names the same item.
pub use cell::Sigma;

/// Typed forwarding wrapper: it fixes the face, key and address types and calls the real
/// `src/table/keyrep.rs` `I64WordCodec::encode` body through the real `WordHome`. It restates no
/// formula. This is the body `proof/extract/lib.rs` `word_i64_codec_encode` names.
pub fn word_i64_codec_encode(sigma: cell::Sigma, key: i64) -> u64 {{
    let home = table::WordHome;
    let address = table::Home::address(&home, sigma, &key);
    <table::keyrep::I64WordCodec as table::keyrep::ExactCodec<i64, table::WordHome>>::encode(
        sigma, &key, &address,
    )
}}

/// Typed forwarding wrapper: it calls the real `src/table/keyrep.rs` `I64WordCodec::decode` body
/// through the real `WordHome`. It restates no formula. This is the body
/// `proof/extract/lib.rs` `word_i64_codec_decode` names.
pub fn word_i64_codec_decode(sigma: cell::Sigma, identity: u64) -> i64 {{
    <table::keyrep::I64WordCodec as table::keyrep::ExactCodec<i64, table::WordHome>>::decode(
        sigma, identity,
    )
}}
'''

def lib_source(src):
    """Absolute `#[path]`s to the real modules; `table` and `table/` vary with the control copy."""
    def p(rel):
        return json.dumps(str(src/rel))
    return LIB_TEMPLATE.format(scalar=p('scalar.rs'), cell=p('cell.rs'),
        coord=p('coord.rs'), table=p('table.rs'))

def stage_source(src, label):
    """Stage the real modules the `#[path]` closure names, with `table.rs` and its `table/` tree
    copied so a control can mutate the copied `keyrep.rs` without touching the working tree."""
    stage = SCRATCH/label/'src'
    stage.mkdir(parents=True)
    for rel in ('scalar.rs','cell.rs','coord.rs','table.rs'):
        shutil.copyfile(src/rel, stage/rel)
    for tree in ('coord','table'):
        (stage/tree).mkdir()
        for f in (src/tree).iterdir():
            if f.is_file():
                shutil.copyfile(f, stage/tree/f.name)
    return stage

def extract(src, label):
    """Create only typed forwarding wrappers and extract the real call closure they reach."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="i64_codec_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(src))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    # The real `src/table.rs` test modules run here, so the mutated copy must still compile and
    # the real suite result is recorded. The control asserts its own Rust-level red below.
    code, rtext = run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],
                      project,label+'-rust-tests', require=(label != 'mutant'))
    llbc = project/'i64_codec_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_i64_codec_encode','--start-from','crate::word_i64_codec_decode',
         '--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated = project/'I64CodecSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    # Disposable absolute source paths appear in the `Source:` comments; name each staged module
    # at its repository-relative path so the committed file is byte-stable across runs.
    for rel in ('scalar.rs','cell.rs','coord.rs','table.rs'):
        generated.write_text(generated.read_text().replace(
            str(src/rel), f'src/{rel}'))
    for tree in ('coord','table'):
        for f in (src/tree).iterdir():
            if f.is_file():
                generated.write_text(generated.read_text().replace(
                    str(src/tree/f.name), f'src/{tree}/{f.name}'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('I64_CODEC_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p = SCRATCH/'lean'
    p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage i64CodecProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib I64Codec where\n  roots := #[`I64CodecSource,`I64CodecResource]\n')
    manifest = json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text())
    manifest['name'] = 'i64CodecProof'
    for dep in manifest['packages']:
        if dep['name'] == 'aeneas':
            dep['dir'] = str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps = p/'.lake/packages'
    deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir():
        (deps/dep.name).symlink_to(dep.resolve(), target_is_directory=True)
    return p

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--prepare', action='store_true')
    args.add_argument('--check', action='store_true')
    args.add_argument('--controls', action='store_true')
    a = args.parse_args()
    src = ROOT/'src'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated = extract(stage_source(src,'stage-fresh'),'fresh')
    dest = ROOT/'proof/resource/I64CodecSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest)
        print('I64_CODEC_EXTRACTION_READY')
        return
    if generated.read_bytes() != dest.read_bytes():
        raise RuntimeError('I64_CODEC_EXTRACTION_DRIFT')
    proof = ROOT/'proof/resource/I64CodecResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0', proof.read_text()):
        raise RuntimeError('I64_CODEC_PROOF_POLICY_REJECT')
    p = prepare_lean()
    shutil.copyfile(dest, p/'I64CodecSource.lean')
    shutil.copyfile(proof, p/'I64CodecResource.lean')
    code, text = run(['lake','build','I64CodecResource'], p, 'fresh-proof')
    if 'I64_CODEC_KERNEL_OK' not in text:
        raise RuntimeError('I64_CODEC_AUDIT_MISSING')
    if a.controls:
        # Real-source mutation: `address_identity` in `src/table/keyrep.rs:307` recomputes
        # `spread(key.cast_unsigned()).id`, the same value `encode` already holds, so the body's
        # `debug_assert_eq!` guard is a tautology. Mutating it to return the *placement* half
        # instead makes the guard compare two different words, which must turn
        # `encode_spec`/`encode_work_constant` red (the `massert` arm no longer decides `true`)
        # and must break `codec_round_trip` and `success_witness`. The copy is in scratch; the
        # working tree is never edited.
        keyrep = src/'table'/'keyrep.rs'
        original = keyrep.read_text()
        needle = '    let id = coord::spread(key.cast_unsigned()).id;\n    let _ = address;\n    id\n'
        if original.count(needle) != 1:
            raise RuntimeError('I64_CODEC_CONTROL_TARGET_MISSING')
        mutant = stage_source(src,'stage-mutant')
        (mutant/'table'/'keyrep.rs').write_text(
            original.replace(needle, needle.replace('let id = coord::spread(key.cast_unsigned()).id;',
                'let id = coord::spread(key.cast_unsigned()).place;')))
        freshmut = extract(mutant,'mutant')
        # Second, independent rejection: the mutated copy also fails the real `src/table.rs` test
        # suite, at the mutated `keyrep.rs:288` `debug_assert_eq!`. The Rust-level red and the
        # Lean-level red below are two distinct consumers of the same seeded defect.
        mutant_rust = [r for r in RECORDS if r['label'] == 'mutant-rust-tests']
        if not mutant_rust or mutant_rust[0]['exit'] == 0:
            raise RuntimeError('I64_CODEC_CONTROL_RUST_NOT_RED')
        mutlog = Path(mutant_rust[0]['log']).read_text(errors='replace')
        if 'keyrep.rs:288' not in mutlog or 'assertion' not in mutlog:
            raise RuntimeError('I64_CODEC_CONTROL_RUST_NOT_ATTRIBUTABLE '+mutlog[-1500:])
        print('I64_CODEC_REAL_SOURCE_CONTROL_RUST_RED')
        shutil.copyfile(freshmut, p/'I64CodecSource.lean')
        code, text = run(['lake','build','I64CodecResource'], p, 'mutant-proof', require=False)
        if code != 1 or 'I64CodecResource.lean:' not in text or not any(
                x in text for x in ['unsolved goals','type mismatch','Tactic']):
            raise RuntimeError('I64_CODEC_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('I64_CODEC_REAL_SOURCE_CONTROL_RED')
        # Restore the real body and prove the extraction returns to the committed one.
        (mutant/'table'/'keyrep.rs').write_text(original)
        restored = extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip = lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '', s, flags=re.S)
        if strip(restored.read_text()) != strip(dest.read_text()):
            raise RuntimeError('I64_CODEC_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest, p/'I64CodecSource.lean')
        run(['lake','build','I64CodecResource'], p, 'restored-proof')
        print('I64_CODEC_REAL_SOURCE_CONTROL_RESTORED')
    print('I64_CODEC_SOURCE_RESOURCE_OK')

if __name__ == '__main__':
    status = 1
    try:
        main()
        status = 0
    finally:
        source_paths = [TOOLS/'target/aeneas/charon-driver', TOOLS/'target/aeneas-model/lean/lean-toolchain',
            TOOLS/'proof/dispatch/lean-toolchain', ROOT/'src/scalar.rs', ROOT/'src/cell.rs',
            ROOT/'src/coord.rs', ROOT/'src/table.rs', ROOT/'src/table/keyrep.rs',
            ROOT/'src/table/state.rs', ROOT/'src/table/locator.rs', ROOT/'src/table/prepared.rs',
            ROOT/'src/table/probe.rs', ROOT/'src/table/resources.rs', ROOT/'src/table/scan.rs',
            ROOT/'src/table/parallel.rs', ROOT/'src/table/partition.rs', ROOT/'src/coord/address.rs',
            ROOT/'src/coord/locator.rs', ROOT/'src/coord/locator_core.rs',
            ROOT/'proof/strip-modeled-axioms.py', ROOT/'proof/resource/I64Codec_gate.py',
            ROOT/'proof/resource/I64CodecSource.lean', ROOT/'proof/resource/I64CodecResource.lean',
            TOOLS/'target/aeneas/charon', TOOLS/'target/aeneas/aeneas',
            TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(
            exit=status, seconds=time.monotonic()-START, budget=540,
            argv=[sys.executable,*sys.argv], lane=LANE, memory_max='12G', swap_max=0, threads=1,
            scratch=str(SCRATCH), deadline=os.environ.get('I64_CODEC_DEADLINE'), records=RECORDS,
            sha256={str(p):digest(p) for p in source_paths if p.is_file()},
            trust=('Aeneas/Charon, compiler, IO/process/dependencies; the work measure counts '
                   'Result steps and 64/128-bit word operations under named premises, never '
                   'cycles, wall clock or GPU time')), indent=2)+'\n')
        shutil.rmtree(SCRATCH, ignore_errors=True)
