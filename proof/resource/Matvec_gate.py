#!/usr/bin/env python3
"""Extract and check the real `word_matvec_fixed` source in a bounded private build.

The extraction root is a private crate that pulls the real engine modules by `#[path]`: no
substitute algorithm, no formula restated. The generated Lean is compared declaration-by-declaration
against the canonical-corpus extraction `proof/ApeironWord.lean`, so a drift between this resource
extraction and the corpus is a rejection, not a silent divergence.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')
if __name__ == '__main__' and os.environ.get('MATVEC_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530','/tmp/opencode/apeiron-lane.lock',
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','MATVEC_IN_LANE=1',
        'MATVEC_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Disposable trees live outside the tmpfs: `/tmp` is a 32G tmpfs at 94% on this box.
OUT = Path(os.environ.get('MATVEC_OUT') or '/home/socol/oc-tmp/matvec-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=OUT))

# The real sources the extraction root pulls in by `#[path]`. Each is copied into the throwaway
# crate; the mutation control edits only the copy, never the working tree.
SOURCES = [
    'src/format/quant.rs', 'src/format/gguf.rs',
    'src/engine/f32_convert.rs', 'src/engine/value.rs', 'src/engine/error.rs',
    'src/engine/dequant.rs', 'src/engine/dequant_lowbit.rs', 'src/engine/matvec.rs',
]

# The generated declarations this gate requires, and the ones it compares against the corpus.
DECLS = [
    'matvec.MatvecError', 'matvec.row_dot_fixed_loop.body', 'matvec.row_dot_fixed_loop',
    'matvec.row_dot_fixed', 'matvec.matvec_fixed_loop.body', 'matvec.matvec_fixed_loop',
    'matvec.matvec_fixed', 'word_matvec_fixed', 'dequant.dequant_row_fixed',
    'quant.block_geom', 'value.round_shift_32',
]

LIB_RS = '''//! Private extraction root for `word_matvec_fixed`: the real engine modules by `#[path]`.
#[path = "../../src/format/gguf.rs"]
pub mod gguf;
#[path = "../../src/format/quant.rs"]
pub mod quant;

/// Mirror of `crate::format` for the extraction unit.
pub mod format {
    pub use super::{gguf, quant};
}

#[path = "../../src/engine/f32_convert.rs"]
pub mod f32_convert;
#[path = "../../src/engine/value.rs"]
pub mod value;
#[path = "../../src/engine/error.rs"]
pub mod error;
#[path = "../../src/engine/dequant.rs"]
pub mod dequant;
#[path = "../../src/engine/matvec.rs"]
pub mod matvec;

/// Mirror of `crate::engine` for the extraction unit.
pub mod engine {
    pub use super::{dequant, error, f32_convert, matvec, value};

    /// The engine refusal type at its real `crate::engine::EngineError` path.
    pub use super::error::EngineError;
    /// The projection refusal at its real `crate::engine::MatvecError` path.
    pub use super::matvec::MatvecError;
}

pub use error::EngineError;
pub use matvec::MatvecError;

/// The real fixed-point projection kernel, rooted at the concrete Q32.32 `i64` word.
pub fn word_matvec_fixed(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    x: &[i64],
) -> Result<Vec<i64>, matvec::MatvecError> {
    matvec::matvec_fixed(ggml_type, raw, n_out, n_in, x)
}
'''

CARGO_TOML = '''[package]
name = "apeiron_word"
version = "0.0.0"
edition = "2021"

[lib]
path = "lib.rs"

[features]
default = []
injection = []
read-stats = []

[workspace]
'''


def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('MATVEC_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('MATVEC_BUDGET_EXHAUSTED')
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


def make_root(root):
    """Create the private extraction crate; `root/src` holds the throwaway source copies."""
    crate = root/'proof/extract'
    (crate/'proof').mkdir(parents=True, exist_ok=True)
    for rel in SOURCES:
        dst = root/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, dst)
    (crate/'Cargo.toml').write_text(CARGO_TOML)
    (crate/'lib.rs').write_text(LIB_RS)


def extract(label, root, source_override=None):
    """Extract the real call closure rooted at `crate::word_matvec_fixed`; return the Lean file."""
    crate = root/'proof/extract'
    outdir = root/'lean-out'
    outdir.mkdir(exist_ok=True)
    llbc = root/'apeiron_word.llbc'
    run(['cargo','generate-lockfile','--offline'], crate, label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(root/'rust-target')],
        crate, label+'-rust-tests', require=(label != 'mutant'))
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas',
         '--dest-file',str(llbc),'--start-from','crate::word_matvec_fixed','--',
         '--manifest-path',str(crate/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(root/'charon-target')], crate, label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(outdir),
         '-warnings-as-errors','-no-progress-bar',str(llbc)], crate, label+'-aeneas')
    generated = outdir/'ApeironWord.lean'
    if not generated.is_file():
        raise RuntimeError('MATVEC_GENERATION_MISSING')
    text = generated.read_text()
    if source_override:
        text = text.replace(str(source_override), source_override.name)
    else:
        text = re.sub(r"Source: '(?:\.\./)*" + 'proof/extract' + r"/lib\.rs'",
                      "Source: 'proof/extract/lib.rs'", text)
        text = re.sub(r"Source: '(?:\.\./)*src/", "Source: 'src/", text)
    # Aeneas emits one `axiom` per external it does not model; the patched model at
    # `target/aeneas-model/lean` defines the ones in its allowlist, and
    # `proof/strip-modeled-axioms.py` drops exactly those stubs. Any remaining `axiom`,
    # `sorry` or `admit` is a real unmodeled external and fails the gate.
    dest = root/'MatvecSource.lean'
    dest.write_text(text)
    stripped = run([sys.executable, str(ROOT/'proof/strip-modeled-axioms.py'), str(dest)],
                   ROOT, label+'-strip')[1]
    if re.search(r'(^|\s)(axiom|sorry|admit)(\s|$)', dest.read_text()):
        raise RuntimeError('MATVEC_UNMODELED_SOURCE '+stripped[-800:])
    # Aeneas leaves blanks at line ends; normalize so a rerun is byte-identical.
    dest.write_text('\n'.join(l.rstrip() for l in dest.read_text().split('\n')))
    return dest


def decl_text(text, name):
    """The exact generated text of one declaration, docstring and attributes included."""
    lines = text.split('\n')
    for i, l in enumerate(lines):
        m = re.match(r'^(?:@\[[^\]]*\]\s*)?(?:private |protected )?'
                     r'(?:def|abbrev|theorem|inductive|structure)\s+' + re.escape(name) + r'\b', l)
        if m:
            start = i
            j = i - 1
            while j >= 0 and (lines[j].startswith('/--') or lines[j].startswith(' *')
                              or lines[j].startswith('@[')):
                if lines[j].startswith('/--') or lines[j].startswith('@['):
                    start = j
                j -= 1
            end = i + 1
            while end < len(lines) and not re.match(
                    r'^(def|abbrev|theorem|inductive|structure|@\[|/--|end |namespace |axiom )', lines[end]):
                end += 1
            return '\n'.join(lines[start:end]).rstrip()
    return None


def compare_with_corpus(fresh_text):
    """Fail-closed: every required declaration must be byte-identical to the corpus extraction."""
    committed = (ROOT/'proof/ApeironWord.lean').read_text()
    bad = []
    for n in DECLS:
        a, b = decl_text(fresh_text, n), decl_text(committed, n)
        if a is None:
            bad.append(n+': MISSING_IN_FRESH'); continue
        if b is None:
            bad.append(n+': MISSING_IN_CORPUS'); continue
        if a != b:
            bad.append(n+': DIFFERS_FROM_CORPUS')
    if bad:
        raise RuntimeError('MATVEC_CORPUS_DRIFT '+'; '.join(bad))
    print('MATVEC_CORPUS_IDENTICAL '+str(len(DECLS)))


def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p = SCRATCH/'lean'
    p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text(
        'import Lake\nopen Lake DSL\npackage matvecProof where\n'
        '  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\n'
        'require aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n'
        '@[default_target]\nlean_lib Matvec where\n  roots := #[`MatvecSource, `MatvecResource]\n')
    manifest = json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text())
    manifest['name'] = 'matvecProof'
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
    root = SCRATCH/'repo'
    make_root(root)
    run(['rustc','--version'], ROOT, 'rustc-version')
    run(['cargo','--version'], ROOT, 'cargo-version')
    run(['lean','--version'], ROOT, 'lean-version')
    generated = extract('fresh', root)
    compare_with_corpus(generated.read_text())
    dest = ROOT/'proof/resource/MatvecSource.lean'
    if a.prepare:
        shutil.copyfile(generated, dest)
        print('MATVEC_EXTRACTION_READY')
        return
    if generated.read_bytes() != dest.read_bytes():
        raise RuntimeError('MATVEC_EXTRACTION_DRIFT')
    proof = ROOT/'proof/resource/MatvecResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0', proof.read_text()):
        raise RuntimeError('MATVEC_PROOF_POLICY_REJECT')
    p = prepare_lean()
    shutil.copyfile(dest, p/'MatvecSource.lean')
    shutil.copyfile(proof, p/'MatvecResource.lean')
    run(['lake','build','MatvecResource'], p, 'fresh-proof')
    print('MATVEC_KERNEL_OK')
    if a.controls:
        # Mutation control on the real source: one copy edited, in a throwaway tree.
        original = (ROOT/'src/engine/matvec.rs').read_text()
        needle = 'while i < w_row.len() && acc_ok {'
        if original.count(needle) != 1:
            raise RuntimeError('MATVEC_CONTROL_TARGET_MISSING')
        mutant_root = SCRATCH/'mutant'
        make_root(mutant_root)
        mutant = mutant_root/'src/engine/matvec.rs'
        mutant.write_text(original.replace(
            needle, 'while i < w_row.len() && acc_ok && false {'))
        freshmut = extract('mutant', mutant_root, source_override=mutant)
        shutil.copyfile(freshmut, p/'MatvecSource.lean')
        code,_ = run(['lake','build','MatvecResource'], p, 'mutant-proof', require=False)
        text = (OUT/(SCRATCH.name+'-mutant-proof.log')).read_text(errors='replace')
        if code == 0 or 'MatvecResource.lean:' not in text or not any(
                x.lower() in text.lower() for x in ['unsolved goals','type mismatch','tactic',
                                                    'declaration uses','unknown constant',
                                                    'motive is not type correct',
                                                    'forbidden axioms']):
            raise RuntimeError('MATVEC_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('MATVEC_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original)
        restored = extract('restored', mutant_root, source_override=mutant)
        strip = lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '', s, flags=re.S)
        if strip(restored.read_text()) != strip(dest.read_text()):
            raise RuntimeError('MATVEC_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest, p/'MatvecSource.lean')
        run(['lake','build','MatvecResource'], p, 'restored-proof')
        print('MATVEC_REAL_SOURCE_CONTROL_RESTORED')
    print('MATVEC_SOURCE_RESOURCE_OK')


if __name__ == '__main__':
    status = 1
    try:
        main()
        status = 0
    finally:
        source_paths = [TOOLS/'target/aeneas/charon-driver', TOOLS/'target/aeneas-model/lean/lean-toolchain',
                        TOOLS/'proof/dispatch/lean-toolchain', ROOT/'proof/ApeironWord.lean',
                        ROOT/'src/engine/matvec.rs', ROOT/'src/engine/dequant.rs',
                        ROOT/'src/engine/value.rs', ROOT/'src/engine/error.rs',
                        ROOT/'src/format/quant.rs', ROOT/'src/format/gguf.rs',
                        ROOT/'proof/resource/Matvec_gate.py', ROOT/'proof/resource/MatvecSource.lean',
                        ROOT/'proof/resource/MatvecResource.lean', TOOLS/'target/aeneas/charon',
                        TOOLS/'target/aeneas/aeneas', TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(
            exit=status, seconds=time.monotonic()-START, budget=540, argv=[sys.executable, *sys.argv],
            lane='/tmp/opencode/apeiron-lane.lock', memory_max='12G', swap_max=0, threads=1,
            deadline=os.environ.get('MATVEC_DEADLINE'), records=RECORDS, decls=DECLS,
            sha256={str(p): digest(p) for p in source_paths if p.is_file()},
            trust='Aeneas/Charon, compiler, IO/process/dependencies; the resource claim counts '
                  'element/row/append units and excludes GPU, heap-allocator and hardware timing'
        ), indent=2)+'\n')
        shutil.rmtree(SCRATCH)