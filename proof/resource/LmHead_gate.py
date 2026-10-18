#!/usr/bin/env python3
"""Extract and check the real LM-head projection source in a bounded private build.

The shared `proof/extract/lib.rs` unit does not compile the current tree, so this gate never
compiles it. Its own scratch manifest wires the real
`src/{format/{quant,gguf},engine/{value,f32_convert,error,dequant,matvec,lm_head}}.rs` modules by
`#[path]`; the only new Rust code is the typed forwarding wrapper that calls the real
`src/engine/ops/lm_head.rs` `lm_head_fixed` body. No substitute algorithm is extracted.

The row-level claim this gate accepts is conditional: `row_body_visit_of_dot_ok` carries the premise
`hdot` that the row's dot computation returns `ok`, and its two last arms enumerate the
dequantizer's own `vis`/`div` propagation. The unconditional row-body-success claim and the executed
row-visit count are retired, declared by the proof file and reported by this gate; `check_claims`
refuses to run the build unless that declaration, the dot premise and the audited claim set are
present, and `--controls` mutates each of the two premises to show it is load-bearing.

O(B) source/tool/log bytes; external compilation is timed, not assigned constant cost.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
if __name__ == '__main__' and os.environ.get('LM_HEAD_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530','/tmp/opencode/apeiron-lane-1.lock',
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','LM_HEAD_IN_LANE=1',
        'LM_HEAD_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (AGENTS.md failure register: /tmp tmpfs headroom); only named
# receipts and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/lmhead-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path(tempfile.gettempdir()) / 'lmhead-resource'
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))

# The real source closure `lm_head.rs` composes: `crate::format::quant`, `super::matvec::matvec_fixed`,
# `super::{dequant::dequant_row_fixed, value::round_shift_32, EngineError}`. Modules are declared at
# the crate root (a `#[path]` inside an inline module resolves against a virtual directory) with
# `format`/`engine` mirrors, exactly the wiring `proof/extract/lib.rs` uses; the engine mirror
# supplies each `super::` import.
LIB_TEMPLATE = '''pub mod format {{
    pub use super::{{gguf, quant}};
}}
pub mod engine {{
    pub use super::{{dequant, error, f32_convert, lm_head, matvec, value}};
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
#[path = {lm_head}]
pub mod lm_head;
pub use error::EngineError;
pub use matvec::MatvecError;

/// Typed forwarding wrapper: it fixes the quant id, payload, geometry and result types and calls
/// the real `src/engine/ops/lm_head.rs` `lm_head_fixed` body. It restates no formula.
pub fn lm_head_forward(
    ggml_type: u32,
    raw: &[u8],
    vocab: usize,
    n_embd: usize,
    x: &[i64],
) -> Result<Vec<i64>, engine::lm_head::LmHeadError> {{
    engine::lm_head::lm_head_fixed(ggml_type, raw, vocab, n_embd, x)
}}
'''

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

# --- The audited claim set, and the claims this gate has retired.
#
# `row_body_visit` asserted that every visited row makes the row-loop body return
# `ok (ControlFlow.cont ...)`, with no premise on the row's dot computation. It was replaced by the
# conditional `row_body_visit_of_dot_ok`, which carries the premise `hdot` below. The gate accepts
# that replacement only under the condition it was made under: the dot premise is stated exactly,
# the output-capacity premise `hout` is still there and still used, the two premises are in the
# mutation controls, and the file declares the retirement of the unconditional claim and of the
# executed row-visit count. Nothing here re-audits a retired claim, and nothing here reports the
# unconditional claim as proved.
DOT_PREMISE = ('(hdot : ∀ (cb rb3 : Logits) (row : Slice U8) (u : Unit),\n'
    '        dequant.dequant_row_fixed ggml_type row n_in cb = ok (core.result.Result.Ok u, rb3) →\n'
    '          ∃ w : I64, matvec.row_dot_fixed (alloc.vec.Vec.deref rb3) x = ok (core.result.Result.Ok w)) :')
HOUT_PREMISE = '    (hout : out.val.length < Usize.max)\n'
HOUT_USE = '                  have := hout; omega)'
RETIREMENT = 'LM_HEAD_RETIRED_CLAIMS row_body_success_unconditional row_visit_count_end_to_end'
REQUIRED_AUDITS = frozenset((
    'LmHeadResource.row_body_capacity', 'LmHeadResource.row_body_visit_of_dot_ok',
    'LmHeadResource.row_body_exit_rows', 'LmHeadResource.row_body_exit_flag',
    'LmHeadResource.row_visits_closed_form', 'LmHeadResource.lm_head_weight_positions',
    'LmHeadResource.dot_body_visit', 'LmHeadResource.dot_body_exit_end',
    'LmHeadResource.dot_body_exit_flag', 'LmHeadResource.dot_body_exhaustive',
    'LmHeadResource.head_delegates_ok', 'LmHeadResource.head_delegates_err',
    'LmHeadResource.head_shape_refusal', 'LmHeadResource.head_quant_refusal',
    'LmHeadResource.head_quant_lookup_ok'))
RETIRED_AUDITS = frozenset((
    'LmHeadResource.row_body_visit', 'LmHeadResource.row_visits_total',
    'LmHeadResource.row_visit_count'))

def check_claims(text):
    """The audit list, the two premises and the retirement declaration, read from the proof file.

    O(B) in the file's bytes. Returns the audited declarations in file order."""
    audits = list(dict.fromkeys(re.findall(r'(?m)^#resource_audit (\S+)$', text)))
    missing = sorted(a for a in REQUIRED_AUDITS if a not in audits)
    if missing: raise RuntimeError('LM_HEAD_AUDIT_CLAIM_MISSING '+str(missing))
    back = sorted(a for a in audits if a in RETIRED_AUDITS)
    if back: raise RuntimeError('LM_HEAD_RETIRED_CLAIM_REAUDITED '+str(back))
    for name, once in (('DOT', DOT_PREMISE), ('HOUT', HOUT_PREMISE), ('HOUT_USE', HOUT_USE)):
        if text.count(once) != 1: raise RuntimeError(f'LM_HEAD_{name}_PREMISE_NOT_EXACT')
    if text.count(RETIREMENT) != 1: raise RuntimeError('LM_HEAD_RETIREMENT_UNDECLARED')
    return audits

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('LM_HEAD_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('LM_HEAD_BUDGET_EXHAUSTED')
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

def lib_source(matvec):
    """Absolute `#[path]`s to the real modules; only the control copy of `matvec` varies."""
    src = ROOT/'src'
    def p(rel):
        return json.dumps(str(src/rel))
    return LIB_TEMPLATE.format(quant=p('format/quant.rs'), gguf=p('format/gguf.rs'),
        value=p('engine/value.rs'), f32_convert=p('engine/f32_convert.rs'),
        error=p('engine/error.rs'), dequant=p('engine/dequant.rs'),
        lm_head=p('engine/lm_head.rs'), matvec=json.dumps(str(matvec)))

def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="lm_head_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'lm_head_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::lm_head_forward','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated = project/'LmHeadSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    text = generated.read_text().replace(str(ROOT)+'/', '')
    for rel in ['format/quant.rs','format/gguf.rs','engine/value.rs','engine/f32_convert.rs',
                'engine/error.rs','engine/dequant.rs','engine/lm_head.rs']:
        text = text.replace(str(ROOT/'src'/rel), 'src/'+rel)
    text = text.replace(str(source), 'src/engine/ops/matvec.rs')
    generated.write_text(text)
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('LM_HEAD_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p = SCRATCH/'lean'; p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage lmHeadProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib LmHead where\n  roots := #[`LmHeadSource,`LmHeadResource]\n')
    manifest = json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text()); manifest['name'] = 'lmHeadProof'
    for dep in manifest['packages']:
        if dep['name'] == 'aeneas': dep['dir'] = str(TOOLS/'target/aeneas-model/lean')
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
    source = ROOT/'src/engine/ops/matvec.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated = extract(source,'fresh')
    dest = ROOT/'proof/resource/LmHeadSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest); print('LM_HEAD_EXTRACTION_READY'); return
    if generated.read_bytes() != dest.read_bytes(): raise RuntimeError('LM_HEAD_EXTRACTION_DRIFT')
    proof = ROOT/'proof/resource/LmHeadResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0', proof.read_text()):
        raise RuntimeError('LM_HEAD_PROOF_POLICY_REJECT')
    claims = check_claims(proof.read_text())
    p = prepare_lean()
    shutil.copyfile(dest, p/'LmHeadSource.lean')
    shutil.copyfile(proof, p/'LmHeadResource.lean')
    code, text = run(['lake','build','LmHeadResource'], p, 'fresh-proof')
    if 'LM_HEAD_KERNEL_OK' not in text: raise RuntimeError('LM_HEAD_AUDIT_MISSING')
    if RETIREMENT not in text: raise RuntimeError('LM_HEAD_RETIREMENT_MARKER_MISSING')
    if a.controls:
        # The two premises the conditional row theorem carries, each a mutation control on the
        # private copy. Reverting `hout` must redden at the `Vec.push` capacity premise, which is
        # where the boundary refusal re-enters the proof; weakening the dot premise to a trivially
        # true `True` must stop the theorem from closing, which is what shows the premise is
        # load-bearing and the claim conditional. The byte-identical restore must build green.
        original_proof = proof.read_text()
        for label, edits, needles in (
            ('hout', ((HOUT_PREMISE, ''), (HOUT_USE, '                  omega)')), ('omega',)),
            ('dot', ((DOT_PREMISE, '(hdot : True) :'),),
                ('Function expected', 'type mismatch', 'Tactic'))):
            mutated = original_proof
            for old, new in edits:
                if mutated.count(old) != 1:
                    raise RuntimeError('LM_HEAD_PROOF_CONTROL_TARGET_MISSING '+label)
                mutated = mutated.replace(old, new)
            (p/'LmHeadResource.lean').write_text(mutated)
            code, mtext = run(['lake','build','LmHeadResource'], p, 'mutant-'+label+'-proof',
                              require=False)
            if code != 1 or 'LmHeadResource.lean:' not in mtext or not any(n in mtext for n in needles):
                raise RuntimeError('LM_HEAD_PROOF_CONTROL_NOT_ATTRIBUTABLE '+label+'\n'+mtext[-1800:])
            print('LM_HEAD_'+label.upper()+'_CONTROL_RED')
        (p/'LmHeadResource.lean').write_text(original_proof)
        if (p/'LmHeadResource.lean').read_bytes() != proof.read_bytes():
            raise RuntimeError('LM_HEAD_PROOF_RESTORE_DRIFT')
        run(['lake','build','LmHeadResource'], p, 'restored-proof')
        print('LM_HEAD_PROOF_CONTROLS_RESTORED')
        # The row-loop advance in the real composed projection `src/engine/ops/matvec.rs`. The extracted
        # root is the LM head, whose row loop is this loop; the control makes the row count per row
        # of work wrong and must turn the resource measure red.
        original = source.read_text(); needle = 'o += 1;'
        if original.count(needle) != 1: raise RuntimeError('LM_HEAD_CONTROL_TARGET_MISSING')
        mutant = SCRATCH/'matvec.rs'; mutant.write_text(original.replace(needle, 'o += 2;'))
        freshmut = extract(mutant,'mutant'); shutil.copyfile(freshmut, p/'LmHeadSource.lean')
        code, text = run(['lake','build','LmHeadResource'], p, 'mutant-proof', require=False)
        if code != 1 or 'LmHeadResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic','declaration uses']):
            raise RuntimeError('LM_HEAD_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('LM_HEAD_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original); restored = extract(mutant,'restored')
        # The `Source:` comments and the order in which Aeneas emits the declarations both follow
        # the disposable absolute source path; compare the declaration bodies as a multiset.
        strip = lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '', s, flags=re.S)
        chunks = lambda s: sorted(c.strip() for c in re.split(r'\n\s*\n', strip(s)) if c.strip())
        if chunks(restored.read_text()) != chunks(dest.read_text()): raise RuntimeError('LM_HEAD_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest, p/'LmHeadSource.lean'); run(['lake','build','LmHeadResource'], p, 'restored-proof')
        print('LM_HEAD_REAL_SOURCE_CONTROL_RESTORED')
    print('LM_HEAD_AUDIT_LIST '+' '.join(claims))
    print(RETIREMENT)
    print('LM_HEAD_SOURCE_RESOURCE_OK')

if __name__ == '__main__':
    status = 1
    try:
        main(); status = 0
    finally:
        source_paths = [TOOLS/'target/aeneas/charon-driver', TOOLS/'target/aeneas-model/lean/lean-toolchain',
            TOOLS/'proof/dispatch/lean-toolchain', ROOT/'src/engine/ops/matvec.rs', ROOT/'src/engine/ops/lm_head.rs',
            ROOT/'proof/strip-modeled-axioms.py', ROOT/'proof/resource/LmHead_gate.py',
            ROOT/'proof/resource/LmHeadSource.lean', ROOT/'proof/resource/LmHeadResource.lean',
            TOOLS/'target/aeneas/charon', TOOLS/'target/aeneas/aeneas', TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane='/tmp/opencode/apeiron-lane-1.lock',memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('LM_HEAD_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; scalar visit-count/resource claim excludes GPU/hardware timing and the unextracted read adapter'),indent=2)+'\n')
        shutil.rmtree(SCRATCH, ignore_errors=True)