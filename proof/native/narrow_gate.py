#!/usr/bin/env python3
"""Fresh real std narrowing, local error representation, proof and rejecting controls.

SCOPE: NarrowSource.lean, NarrowRefinement.lean, narrow_gate.py.
REGISTER: actual scalar extraction/refinement; excludes shared Unit helpers and GPU/cost.
CORPUS: pinned core conversion/error types and Aeneas IScalar.cast_inBounds_spec.
REFERENCE: /tmp/opencode/native-narrow-bridge-review.md, corrected Luna verdict.
GATE: python3 proof/native/narrow_gate.py --controls, 540s including descendants/lane wait.
GRADE: pending kernel/proof and attributable shape/lower/upper controls.
"""
from pathlib import Path
import argparse, copy, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
if __name__ == '__main__' and os.environ.get('NARROW_IN_LANE') != '1':
    os.execvp('timeout', ['timeout', '--kill-after=5', '540s', 'flock', '-w', '530',
        '/tmp/opencode/apeiron-lane.lock', 'systemd-run', '--user', '--scope', '-q',
        '-p', 'MemoryMax=12G', '-p', 'MemorySwapMax=0', 'env', 'LEAN_NUM_THREADS=1',
        'NARROW_IN_LANE=1', 'NARROW_STARTED='+str(time.time()),
        'NARROW_DEADLINE='+str(time.time()+530), sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
START = time.monotonic()
OUT = Path('/tmp/opencode/native-narrow-proof')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='check-', dir=OUT))
RECORDS = []
SOURCE_PATHS = []
ENUM = ['Empty', 'InvalidDigit', 'PosOverflow', 'NegOverflow', 'Zero', 'NotAPowerOfTwo']
ALIAS = '\n/-- Reviewed representation: erase the real one-field Rust tuple wrapper.\n    Shared Unit-valued conversion helpers are excluded from this package. -/\nabbrev core.num.error.TryFromIntError := core.num.error.IntErrorKind\n'

def digest(path):
    """O(B) input bytes, O(1) streaming scratch."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """O(L) log bytes; all subprocesses consume the single invocation deadline."""
    left = float(os.environ['NARROW_DEADLINE'])-time.time()-3
    if left <= 0:
        raise RuntimeError('NARROW_BUDGET_EXHAUSTED')
    log = OUT/(SCRATCH.name+'-'+label+'.log')
    begin = time.monotonic()
    with log.open('wb') as stream:
        proc = subprocess.Popen(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'LEAN_NUM_THREADS':'1'}, start_new_session=True)
        try:
            proc.wait(timeout=left)
        except BaseException:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait(); raise
    text = log.read_text(errors='replace')
    RECORDS.append(dict(label=label, argv=argv, cwd=str(cwd), exit=proc.returncode,
        seconds=time.monotonic()-begin, log=str(log)))
    if require and proc.returncode:
        raise RuntimeError(label+'_REJECT '+str(proc.returncode)+'\n'+text[-6000:])
    return proc.returncode, text

def name(item):
    """O(P) path components; tuple/impl paths deliberately do not match type names."""
    return '::'.join(x['Ident'][0] for x in item['item_meta']['name'] if 'Ident' in x)

def adt_refs(value):
    """O(A) AST nodes, O(A) collected references; recursive descent visits each node once."""
    if isinstance(value, dict):
        found = [value['Adt']['id']] if 'Adt' in value and isinstance(value['Adt'], dict) else []
        return found + [x for child in value.values() for x in adt_refs(child)]
    if isinstance(value, list):
        return [x for child in value for x in adt_refs(child)]
    return []

def validate_shape(document):
    """O(A) LLBC nodes/bytes, O(A) scratch: bind the actual one-field error to its enum."""
    if document['has_errors']:
        raise RuntimeError('NARROW_LLBC_HAS_ERRORS')
    translated = document['translated']
    types = {name(t):t for t in translated['type_decls'] if t}
    error = types['core::num::error::TryFromIntError']
    kind = types['core::num::error::IntErrorKind']
    fields = error['kind'].get('Struct', [])
    variants = kind['kind'].get('Enum', [])
    if error['item_meta']['opacity'] != 'Transparent' or len(fields) != 1:
        raise RuntimeError('NARROW_ERROR_STRUCT_SHAPE')
    if not fields[0]['is_positional'] or fields[0]['name'] != '_0' or adt_refs(fields[0]['ty']) != [kind['def_id']]:
        raise RuntimeError('NARROW_ERROR_FIELD_BINDING')
    if kind['item_meta']['opacity'] != 'Transparent' or [v['name'] for v in variants] != ENUM or any(v['fields'] for v in variants):
        raise RuntimeError('NARROW_ERROR_ENUM_SHAPE')
    wrapper = next(f for f in translated['fun_decls'] if f and name(f)=='native_narrow::narrow')
    signature = wrapper['signature']
    input_id, input_type = signature['inputs'][0]['Value']
    output_id, output_type = signature['output']['Value']
    if input_type != {'Scalar':{'Integer':{'Signed':'I128'}}} or output_type['Adt']['generics']['types'][0]['Value'][1] != {'Scalar':{'Integer':{'Signed':'I64'}}} or adt_refs(output_type['Adt']['generics']['types'][1]) != [error['def_id']]:
        raise RuntimeError('NARROW_WRAPPER_SIGNATURE')
    signed = [f for f in translated['fun_decls'] if f and name(f)=='core::convert::num::try_from'
        and '/convert/num.rs' in str(translated['files'][f['item_meta']['span']['Untagged']['data']['file_id']])
        and f.get('body') is not None and f['signature']['inputs']==[{'Deduplicated':input_id}]
        and f['signature']['output']=={'Deduplicated':output_id}]
    if len(signed)!=1:
        raise RuntimeError('NARROW_CONVERSION_BODY_MISSING')

def bridge(text):
    """O(B) generated bytes, O(B) result; one insertion, no conversion operation rewrite."""
    if re.search(r'\b(axiom|sorry|admit|noncomputable)\b', text):
        raise RuntimeError('NARROW_GENERATED_POLICY')
    if text.count('namespace native_narrow') != 1 or text.count('end native_narrow') != 1:
        raise RuntimeError('NARROW_NAMESPACE')
    anchor = '| NotAPowerOfTwo : core.num.error.IntErrorKind\n'
    if text.count(anchor) != 1 or [x for x in re.findall(r'^\| (\w+) : core.num.error.IntErrorKind$', text, re.M)] != ENUM:
        raise RuntimeError('NARROW_GENERATED_ENUM')
    if text.count('def I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from') != 1:
        raise RuntimeError('NARROW_GENERATED_CONVERSION')
    for payload in ['Err core.num.error.IntErrorKind.NegOverflow', 'Err core.num.error.IntErrorKind.PosOverflow']:
        if text.count(payload) != 1:
            raise RuntimeError('NARROW_ERROR_PAYLOAD')
    adapted = text.replace(anchor, anchor+ALIAS)
    if adapted.replace(ALIAS, '') != text:
        raise RuntimeError('NARROW_BODY_REWRITE')
    return adapted

def prepare_lake():
    """O(D) dependency links, no shared build mutations; uses existing pinned artifacts."""
    project = SCRATCH/'lean'; project.mkdir()
    (project/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (project/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage narrowProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib Narrow where\n  roots := #[`NarrowSource,`NarrowRefinement]\n')
    manifest = json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text()); manifest['name']='narrowProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (project/'lake-manifest.json').write_text(json.dumps(manifest))
    dependencies = project/'.lake/packages'; dependencies.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir():
        (dependencies/dep.name).symlink_to(dep.resolve(), target_is_directory=True)
    return project

def main():
    """O(source/LLBC/log bytes); fresh extraction and all controls share one bounded gate."""
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--controls',action='store_true');args=parser.parse_args()
    toolchain=run([str(TOOLS/'target/aeneas/charon'),'toolchain-path'], ROOT, 'charon-sysroot')[1].strip()
    sysroot=Path(toolchain)
    for argv,label in [(['rustc','--version'],'shell-rustc'),(['cargo','--version'],'cargo'),
        ([str(sysroot/'bin/rustc'),'--version'],'selected-rustc'),([str(sysroot/'bin/rustc'),'--print','sysroot'],'selected-sysroot'),
        ([str(TOOLS/'target/aeneas/charon'),'version'],'charon-version'),([str(TOOLS/'target/aeneas/aeneas'),'-version'],'aeneas-version')]:
        run(argv,ROOT,label)
    error_source=sysroot/'lib/rustlib/src/rust/library/core/src/num/error.rs'
    convert_source=sysroot/'lib/rustlib/src/rust/library/core/src/convert/num.rs'
    SOURCE_PATHS.extend([error_source,convert_source])
    if digest(error_source)!='f66e9b263eb9f9aab957801d08a4a8185d7d15e5945a9f491023986d19ca2636' or digest(convert_source)!='cef1248e8e76d58077b34f622c51d07f25d78a1293783a6bd7ae371f639b84db':
        raise RuntimeError('NARROW_REVIEWED_SOURCE_DRIFT')
    rust=SCRATCH/'rust';rust.mkdir()
    (rust/'Cargo.toml').write_text('[package]\nname="native_narrow"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[workspace]\n')
    (rust/'lib.rs').write_text('pub fn narrow(x:i128)->Result<i64,core::num::TryFromIntError>{i64::try_from(x)}\n')
    run(['cargo','generate-lockfile','--offline'],rust,'lock')
    llbc=rust/'NarrowSource.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
        '--start-from','crate::narrow','--include','core::convert::num::_','--include','core::num::error::TryFromIntError',
        '--include','core::num::error::IntErrorKind','--','--manifest-path',str(rust/'Cargo.toml'),'--locked','--offline','--lib',
        '--target-dir',str(SCRATCH/'rust-target')],rust,'fresh-charon')
    document=json.loads(llbc.read_text());validate_shape(document)
    if args.controls:
        corrupted=copy.deepcopy(document)
        t=next(x for x in corrupted['translated']['type_decls'] if x and name(x)=='core::num::error::TryFromIntError')
        t['kind']['Struct'][0]['ty']={'Adt':{'id':999999}}
        (SCRATCH/'corrupt-shape.llbc').write_text(json.dumps(corrupted))
        try: validate_shape(corrupted)
        except RuntimeError as error:
            if str(error)!='NARROW_ERROR_FIELD_BINDING':raise
            print('NARROW_SHAPE_CONTROL_RED',flush=True)
        else:raise RuntimeError('NARROW_SHAPE_CONTROL_SURVIVED')
        validate_shape(document);print('NARROW_SHAPE_CONTROL_RESTORED',flush=True)
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(rust),'-warnings-as-errors','-all-computable','-no-progress-bar',str(llbc)],rust,'fresh-aeneas')
    raw=(rust/'NarrowSource.lean').read_text();adapted=bridge(raw)
    (SCRATCH/'unadapted.lean').write_text(raw)
    dest=ROOT/'proof/native/NarrowSource.lean'
    if args.prepare:dest.write_text(adapted)
    elif not dest.is_file() or dest.read_text()!=adapted:raise RuntimeError('NARROW_EXTRACTION_DRIFT')
    project=prepare_lake();(project/'NarrowSource.lean').write_text(adapted)
    run(['lake','env','lean','--version'],project,'lean-version')
    run(['git','rev-parse','HEAD'],TOOLS/'proof/dispatch/.lake/packages/mathlib','mathlib-revision')
    SOURCE_PATHS.extend([llbc,rust/'NarrowSource.lean',SCRATCH/'unadapted.lean',rust/'lib.rs'])
    run(['lake','build','NarrowSource'],project,'fresh-source')
    if args.prepare:print('NARROW_SOURCE_READY',flush=True);return
    proof=ROOT/'proof/native/NarrowRefinement.lean'
    if not proof.is_file():raise RuntimeError('NARROW_REFINEMENT_MISSING')
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('NARROW_PROOF_POLICY')
    shutil.copyfile(proof,project/'NarrowRefinement.lean')
    _,text=run(['lake','build','NarrowRefinement'],project,'fresh-proof')
    if 'NARROW_KERNEL_OK' not in text:raise RuntimeError('NARROW_AUDIT_MISSING')
    if args.controls:
        for label,needle,mutation in [('lower','if u < (-9223372036854775808)#i128','if u <= (-9223372036854775808)#i128'),
            ('upper','if u > 9223372036854775807#i128','if u >= 9223372036854775807#i128')]:
            if adapted.count(needle)!=1:raise RuntimeError('NARROW_GUARD_TARGET_MISSING')
            (project/'NarrowSource.lean').write_text(adapted.replace(needle,mutation))
            run(['lake','build','NarrowSource'],project,label+'-mutant-source')
            code,text=run(['lake','build','NarrowRefinement'],project,label+'-mutant-proof',require=False)
            boundary = 'NarrowRefinement.'+label+'_boundary: forbidden dependencies [sorryAx]'
            if code!=1 or 'NarrowRefinement.lean:' not in text or 'Tactic `rfl` failed' not in text or boundary not in text:
                raise RuntimeError('NARROW_CONTROL_NOT_ATTRIBUTABLE '+label+'\n'+text[-2000:])
            print('NARROW_'+label.upper()+'_CONTROL_RED',flush=True)
            (project/'NarrowSource.lean').write_text(adapted)
            run(['lake','build','NarrowRefinement'],project,label+'-restored-proof')
            print('NARROW_'+label.upper()+'_CONTROL_RESTORED',flush=True)
    print('NARROW_SOURCE_REFINEMENT_OK',flush=True)

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        paths=SOURCE_PATHS+[ROOT/'proof/native/narrow_gate.py',ROOT/'proof/native/NarrowSource.lean',ROOT/'proof/native/NarrowRefinement.lean',
            TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas/aeneas',
            TOOLS/'target/aeneas-model/lean/Aeneas/Std/Scalar/CoreConvertNum.lean',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,
            argv=[sys.executable,*sys.argv],records=RECORDS,sha256={str(p):digest(p) for p in paths if p.is_file()},
            lane='/tmp/opencode/apeiron-lane.lock',memory_max='12G',swap_max=0,threads=1,
            deadline=os.environ.get('NARROW_DEADLINE'),trust='direct pinned std signed conversion; shared Unit helpers, GPU, codegen and hardware/cost excluded'),indent=2)+'\n')
        # Scratch source/LLBC and logs are evidence, retained for reviewer inspection.
