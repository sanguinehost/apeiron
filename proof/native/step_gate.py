#!/usr/bin/env python3
"""Private pinned Step compatibility; complete fresh source gate and controls.
SCOPE: step_compat.patch, StepRefinement.lean, step_gate.py.
REGISTER: actual extraction, model Step wiring and exhausted-tail narrowing; cost excluded.
CORPUS: actual pinned Iter Step fields and existing scalar overflowing operations.
REFERENCE: /tmp/opencode/native-step-compatibility-review.md; generator-route review.
GATE: --actual-exact-dot --controls; one540s invocation incl lane/descendants,12G/no swap/thread1.
GRADE: pending exact positive, attributed registry mutation/restoration and closed audit.
"""
from pathlib import Path
import argparse, copy, hashlib, json, os, re, signal, subprocess, sys, tempfile, time
ROOT=Path(__file__).resolve().parents[2]
MAIN=Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
OUT=Path('/tmp/opencode/native-step-compatibility')
if __name__=='__main__' and os.environ.get('STEP_IN_LANE')!='1':
    now=time.time()
    os.execvp('timeout',['timeout','--kill-after=5','540s','flock','-w','530','/tmp/opencode/apeiron-lane.lock',
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0','env',
        'LEAN_NUM_THREADS=1','STEP_IN_LANE=1','STEP_START='+str(now),'STEP_DEADLINE='+str(now+532),
        sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
DEADLINE=float(os.environ.get('STEP_DEADLINE',time.time()+532))
RECORDS=[]; HASHES={}; ARTIFACTS=[]; WORK=None

def sha(path):
    """O(B) bytes and O(1) scratch, streaming digest."""
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def bind(path):
    if Path(path).is_file():
        digest=sha(path);HASHES[str(path)]=digest
        ARTIFACTS.append(dict(path=str(path),sha256=digest,stage=RECORDS[-1]['stage'] if RECORDS else 'input'))

def run(argv,cwd,label,expected=0,env=None):
    """O(L) captured log bytes; every child consumes the common gate deadline."""
    left=DEADLINE-time.time()-3
    if left<=0:raise RuntimeError('STEP_BUDGET_EXHAUSTED')
    log=OUT/(WORK.name+'-'+label+'.log'); begin=time.monotonic(); argv=list(map(str,argv))
    with log.open('wb') as stream:
        p=subprocess.Popen(argv,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,**(env or {}),'LEAN_NUM_THREADS':'1'},start_new_session=True)
        interrupted=None
        try:p.wait(timeout=left)
        except BaseException as error:
            os.killpg(p.pid,signal.SIGKILL);p.wait();interrupted=error
    text=log.read_text(errors='replace')
    RECORDS.append(dict(stage=label,exit=p.returncode,seconds=time.monotonic()-begin,argv=argv,cwd=str(cwd),log=str(log)))
    print('STEP_STAGE',label,'exit='+str(p.returncode),f'seconds={time.monotonic()-begin:.3f}',str(log),flush=True)
    if interrupted is not None:raise interrupted
    if expected is not None and p.returncode!=expected:raise RuntimeError(label+'_FAILED '+str(p.returncode)+'\n'+text[-7000:])
    return p.returncode,text

def cow(src,dst,label):
    """O(F) file metadata/extent references; private COW cache preserves shared inputs."""
    dst.parent.mkdir(parents=True,exist_ok=True)
    run(['cp','-a','--reflink=always',src,dst],ROOT,label)

def archive(src,commit,tree,dest,label):
    """O(B) archived source bytes; selected immutable commit is checked against its tree."""
    got=run(['git','-C',src,'rev-parse',commit+'^{tree}'],ROOT,label+'-tree')[1].strip()
    if got!=tree:raise RuntimeError('STEP_TREE_DRIFT '+label)
    tar=WORK/(label+'.tar')
    run(['git','-C',src,'archive','--format=tar','-o',tar,commit],ROOT,label+'-archive')
    bind(tar);dest.mkdir(parents=True)
    run(['tar','-xf',tar,'-C',dest],ROOT,label+'-unpack')

def patch(cwd,path,label):
    run(['patch','-p1','--dry-run','-i',path],cwd,label+'-check')
    run(['patch','-p1','-i',path],cwd,label)

def check_source(llbc):
    """O(A) LLBC JSON nodes; exact production root/body/source identity must survive extraction."""
    d=json.loads(llbc.read_text());t=d['translated']
    if d['has_errors']:raise RuntimeError('STEP_CHARON_HAS_ERRORS')
    fs=[f for f in t['fun_decls'] if f and '::'.join(x['Ident'][0] for x in f['item_meta']['name'] if 'Ident' in x)=='apeiron::engine::kern::gemm::exact_dot']
    if len(fs)!=1 or fs[0]['item_meta']['opacity']!='Transparent' or fs[0].get('body') is None:raise RuntimeError('STEP_EXACT_DOT_OPAQUE')
    span=fs[0]['item_meta']['span']['Untagged']['data']
    if 'src/engine/kern/gemm.rs' not in str(t['files'][span['file_id']]):raise RuntimeError('STEP_EXACT_DOT_LOCATION')

def translate(binary,llbc,dest,label,red=False):
    dest.mkdir(exist_ok=True)
    code,text=run([binary,'-backend','lean','-dest',dest,'-warnings-as-errors','-all-computable','-no-progress-bar',llbc],ROOT,label,expected=None)
    if red:
        if code!=2 or "trait decl 'core::iter::range::Step'" not in text or "could not find the information for item 'forward_overflowing'" not in text:
            raise RuntimeError(label+'_UNATTRIBUTED '+str(code)+'\n'+text[-5000:])
    elif code!=0:raise RuntimeError('STEP_NEXT_SOURCE_CLOSURE '+label+' exit='+str(code)+'\n'+text[-7000:])
    else:
        files=list(dest.glob('*.lean'))
        if not files or not any('def engine.kern.gemm.exact_dot' in p.read_text() for p in files):raise RuntimeError('STEP_TRANSLATED_BODY_MISSING')
        for p in files:
            if re.search(r'\b(axiom|sorry|admit)\b',p.read_text()):raise RuntimeError('STEP_GENERATED_POLICY '+str(p))
            bind(p)


ENUM=['Empty','InvalidDigit','PosOverflow','NegOverflow','Zero','NotAPowerOfTwo']
ALIAS='\n/-- Reviewed local representation: erase the actual one-field Rust tuple wrapper. -/\nabbrev core.num.error.TryFromIntError := core.num.error.IntErrorKind\n'
GENERATED_AUDIT=r"""
open Lean Elab Command in
elab "#generated_audit" : command => do
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let environment ← getEnv
  let names := environment.constants.fold (fun acc constant _ =>
    if (`apeiron).isPrefixOf constant then constant :: acc else acc) ([] : List Lean.Name)
  if names.length < 15 then throwError "STEP_GENERATED_AUDIT_POPULATION"
  for constant in names do
    let axioms ← liftTermElabM <| Lean.collectAxioms constant
    let bad := axioms.filter (fun value => !allowed.contains value)
    unless bad.isEmpty do throwError m!"{constant}: forbidden dependencies {bad.toList}"
    logInfo m!"{constant}: dependencies {axioms.toList}"
  logInfo m!"STEP_GENERATED_KERNEL_OK declarations={names.length}"
#generated_audit
"""

def nodes(value):
    """O(A) JSON nodes, O(depth) iterator stack."""
    yield value
    if isinstance(value,dict):
        for child in value.values():yield from nodes(child)
    elif isinstance(value,list):
        for child in value:yield from nodes(child)

def item_name(item):
    return '::'.join(x['Ident'][0] for x in item['item_meta']['name'] if 'Ident' in x)

def validate_narrow(document):
    """O(A) LLBC bytes: actual signed signature, call edge and exact error field/enum."""
    t=document['translated'];types={item_name(x):x for x in t['type_decls'] if x}
    error=types['core::num::error::TryFromIntError'];kind=types['core::num::error::IntErrorKind']
    fields=error['kind'].get('Struct',[])
    if error['item_meta']['opacity']!='Transparent' or len(fields)!=1:raise RuntimeError('STEP_ERROR_STRUCT_SHAPE')
    refs=[x['Adt']['id'] for x in nodes(fields[0]['ty']) if isinstance(x,dict) and 'Adt' in x]
    if not fields[0]['is_positional'] or fields[0]['name']!='_0' or refs!=[kind['def_id']]:raise RuntimeError('STEP_ERROR_FIELD_BINDING')
    variants=kind['kind'].get('Enum',[])
    if kind['item_meta']['opacity']!='Transparent' or [v['name'] for v in variants]!=ENUM or any(v['fields'] for v in variants):raise RuntimeError('STEP_ERROR_ENUM_SHAPE')
    lookup={}
    for x in nodes(t):
        if isinstance(x,dict) and 'Value' in x and isinstance(x['Value'],list) and len(x['Value'])==2:
            index,value=x['Value']
            if isinstance(value,dict) and set(value)&{'Scalar','Adt','Ref','RawPtr','Tuple','Never','TypeVar'}:
                lookup[index]=value
    def resolve(ty):
        if 'Value' in ty:return ty['Value'][1]
        return lookup[ty['Deduplicated']]
    fs=[x for x in t['fun_decls'] if x]
    signed=[f for f in fs if item_name(f)=='core::convert::num::try_from' and f['item_meta']['opacity']=='Transparent']
    if len(signed)!=1:raise RuntimeError('STEP_NARROW_BODY_POPULATION')
    f=signed[0];sig=f['signature']
    if f.get('body') is None or len(sig['inputs'])!=1 or resolve(sig['inputs'][0])!={'Scalar':{'Integer':{'Signed':'I128'}}}:raise RuntimeError('STEP_NARROW_SIGNED_INPUT')
    result=resolve(sig['output'])['Adt'];args=result['generics']['types']
    if item_name(types.get('core::result::Result',{}))!='core::result::Result' or result['id']!=types['core::result::Result']['def_id'] or len(args)!=2 or resolve(args[0])!={'Scalar':{'Integer':{'Signed':'I64'}}} or resolve(args[1])['Adt']['id']!=error['def_id']:raise RuntimeError('STEP_NARROW_SIGNED_OUTPUT')
    location=str(t['files'][f['item_meta']['span']['Untagged']['data']['file_id']])
    if '/convert/num.rs' not in location:raise RuntimeError('STEP_NARROW_SYSROOT_LOCATION')
    for g in fs:
        if item_name(g).startswith('core::convert::num::') and item_name(g).endswith('::from') and g['item_meta']['opacity']=='Transparent':raise RuntimeError('STEP_FROM_MUST_STAY_OPAQUE')
    root=next(g for g in fs if item_name(g)=='apeiron::engine::kern::gemm::exact_dot')
    calls=[n['Regular']['kind'].get('Fun') for n in nodes(root['body']) if isinstance(n,dict) and 'Regular' in n and isinstance(n['Regular'],dict) and 'kind' in n['Regular']]
    if f['def_id'] not in calls:raise RuntimeError('STEP_NARROW_ACTUAL_CALL_EDGE')

def shape_controls(llbc):
    original=json.loads(llbc.read_text())
    for label,expected in [('field','STEP_ERROR_FIELD_BINDING'),('enum','STEP_ERROR_ENUM_SHAPE')]:
        d=copy.deepcopy(original)
        t=next(t for t in d['translated']['type_decls'] if t and item_name(t)=='core::num::error::'+('TryFromIntError' if label=='field' else 'IntErrorKind'))
        if label=='field':t['kind']['Struct'][0]['ty']={'Adt':{'id':999999}}
        else:t['kind']['Enum'].pop()
        path=WORK/(label+'-mutated.llbc');path.write_text(json.dumps(d));bind(path)
        try:validate_narrow(d)
        except RuntimeError as e:
            if str(e)!=expected:raise
            RECORDS.append(dict(stage='llbc-'+label+'-control',exit=1,expected=expected,diagnostic=str(e),argv=['internal-validate',str(path)],cwd=str(ROOT),seconds=0))
        else:raise RuntimeError('STEP_LLBC_CONTROL_SURVIVED '+label)
        validate_narrow(original)
        print('STEP_LLBC_CONTROL_RED_RESTORED',label,flush=True)

def conversion(text):
    match=re.search(r'^def I64\.Insts\.CoreConvertTryFromI128TryFromIntError\.try_from\b.*?(?=\n/\*\*|\n/--|\ndef |\nend )',text,re.M|re.S)
    if not match:raise RuntimeError('STEP_GENERATED_NARROW_MISSING')
    return match.group(0).strip()

def bridge(raw):
    """O(B) bytes: local alias only; compare real conversion operations byte for byte."""
    actual=conversion(raw);accepted=conversion((MAIN/'proof/native/NarrowSource.lean').read_text())
    if actual!=accepted:raise RuntimeError('STEP_ACCEPTED_CONVERSION_DRIFT')
    anchor='| NotAPowerOfTwo : core.num.error.IntErrorKind\n'
    if raw.count(anchor)!=1 or [x for x in re.findall(r'^\| (\w+) : core.num.error.IntErrorKind$',raw,re.M)]!=ENUM:raise RuntimeError('STEP_GENERATED_ENUM_DRIFT')
    adapted=raw.replace(anchor,anchor+ALIAS)
    if adapted.replace(ALIAS,'')!=raw:raise RuntimeError('STEP_ALIAS_OPERATION_REWRITE')
    comparison=WORK/'narrow-comparison.json';comparison.write_text(json.dumps(dict(definition_byte_exact=True,definition_sha256=hashlib.sha256(actual.encode()).hexdigest(),alias_only=True,accepted=str(MAIN/'proof/native/NarrowSource.lean'))));bind(comparison)
    return adapted+GENERATED_AUDIT

def source_compile(lean,label):
    bind(lean/'ExactDotSource.lean')
    _,text=run(['lake','env','lean','-o','.lake/build/lib/lean/ExactDotSource.olean','ExactDotSource.lean'],lean,label)
    if 'STEP_GENERATED_KERNEL_OK' not in text:raise RuntimeError('STEP_GENERATED_AUDIT_MISSING')

def proof_check(lean,label,red=None):
    bind(lean/'StepRefinement.lean')
    code,text=run(['lake','env','lean','-DautoImplicit=false','StepRefinement.lean'],lean,label,expected=None if red else 0)
    if red:
        proof=(lean/'StepRefinement.lean').read_text();start=proof[:proof.index('theorem '+red+' ')].count('\n')+1
        after=proof.index('theorem '+red+' ');next_decl=re.search(r'\n(?:theorem|end) ',proof[after+1:])
        end=proof[:after+1+next_decl.start()].count('\n')+1 if next_decl else len(proof.splitlines())
        diagnostics=[(int(line),message) for line,message in re.findall(r'StepRefinement\.lean:(\d+):\d+: error: ([^\n]+)',text)]
        attributable=any(start<=line<=end and ('Tactic `rfl` failed' in message or 'unsolved goals' in message) for line,message in diagnostics)
        if code!=1 or not attributable or 'StepRefinement.'+red+': forbidden dependencies [sorryAx]' not in text:raise RuntimeError('STEP_SEMANTIC_CONTROL_UNATTRIBUTED '+label+'\n'+text[-6000:])
    elif 'STEP_MODEL_KERNEL_OK' not in text:raise RuntimeError('STEP_AUDIT_MARKER_MISSING')

def semantic_controls(lean,adapted):
    source=lean/'ExactDotSource.lean'
    for label,needle,mutation in [('lower','if u < (-9223372036854775808)#i128','if u <= (-9223372036854775808)#i128'),('upper','if u > 9223372036854775807#i128','if u >= 9223372036854775807#i128')]:
        if adapted.count(needle)!=1:raise RuntimeError('STEP_NARROW_GUARD_TARGET')
        source.write_text(adapted.replace(needle,mutation));source_compile(lean,label+'-guard-source')
        proof_check(lean,label+'-guard-proof',red='tail_'+label+'_boundary')
        source.write_text(adapted);source_compile(lean,label+'-guard-restored-source');proof_check(lean,label+'-guard-restored-proof')
    model=lean/'Aeneas/Std/Core/CoreResult.lean';original=model.read_text();bind(model)
    needle='| .Err _ => _root_.Aeneas.Std.Result.ok none'
    if original.count(needle)!=1:raise RuntimeError('STEP_RESULT_OK_TARGET')
    model.write_text(original.replace(needle,'| .Err _ => _root_.Aeneas.Std.Result.fail .panic'));bind(model)
    run(['lake','build','Aeneas:leanArts'],lean,'result-ok-mutant-model')
    source_compile(lean,'result-ok-mutant-full-source')
    proof_check(lean,'result-ok-mutant-proof',red='tail_negative_refusal')
    model.write_text(original)
    if model.read_text()!=original:raise RuntimeError('STEP_RESULT_OK_RESTORE_DRIFT')
    run(['lake','build','Aeneas:leanArts'],lean,'result-ok-restored-model');bind(model)
    source_compile(lean,'result-ok-restored-full-source');proof_check(lean,'result-ok-restored-proof')


def ocaml_sources(a,c):
    """O(B) private source bytes; bind the exact sources behind the current Dune checkpoint."""
    manifest={}
    for prefix,base in [('aeneas',a/'src'),('charon',c/'charon-ml')]:
        for p in base.rglob('*'):
            relative=p.relative_to(base)
            if '_build' in relative.parts or not p.is_file():continue
            if p.suffix not in {'.ml','.mli','.opam'} and p.name not in {'dune','dune-project','dune-workspace'}:continue
            manifest[prefix+'/'+str(relative)]=sha(p)
    if not manifest or 'aeneas/extract/ExtractBuiltinLean.ml' not in manifest:raise RuntimeError('STEP_TOOL_SOURCE_MANIFEST_MISSING')
    return manifest

def main():
    global WORK
    parser=argparse.ArgumentParser();parser.add_argument('--actual-exact-dot',action='store_true');parser.add_argument('--controls',action='store_true');args=parser.parse_args()
    if not args.actual_exact_dot or not args.controls:raise RuntimeError('STEP_FULL_ACTUAL_CONTROLS_REQUIRED')
    OUT.mkdir(parents=True,exist_ok=True);(ROOT/'target').mkdir(exist_ok=True)
    WORK=Path(tempfile.mkdtemp(prefix='step-sol-',dir=ROOT/'target'));print('STEP_WORK',WORK,flush=True)
    pins=ROOT/'proof/patches/PINS';s=pins.read_text()
    selected={}
    for name in ['AENEAS','CHARON']:
        for key in ['COMMIT','TREE']:selected[name+'_'+key]=re.search(r'^'+name+'_'+key+r'=(\w+)$',s,re.M).group(1)
    # The installed translator is identified by the one authoritative pin file, not by a second
    # literal here; `PINS` content itself is fixed below, so this cannot be relaxed from either side.
    selected['AENEAS_TRANSLATOR_SHA256']=re.search(r'^AENEAS_TRANSLATOR_SHA256=(\w+)$',s,re.M).group(1)
    if selected!={'AENEAS_COMMIT':'505b6ca35217e7be5c96c3e2f8045edfbdf47291','AENEAS_TREE':'af23d5d9d866050c2158700cedab07501443bcdd','CHARON_COMMIT':'b104e24fea7d721b71e6c39fd70f26ff20bc0980','CHARON_TREE':'784a177101bbfd956047ac2ee1b7335d4c057b7b','AENEAS_TRANSLATOR_SHA256':'84f9c1d2e54d866ffc1b0c41091a47cc4ae189cc8671e9f1a3991c5b2610fcd1'}:raise RuntimeError('STEP_PIN_DRIFT')
    for p in [pins,*ROOT.joinpath('proof/native').glob('StepRefinement.lean'),ROOT/'proof/native/step_gate.py',ROOT/'proof/native/step_compat.patch',*ROOT.joinpath('proof/patches').glob('*.patch'),MAIN/'src/lib.rs',MAIN/'src/engine/kern/gemm.rs',MAIN/'src/engine/chart/mod.rs',MAIN/'src/engine/kern/pow2.rs',MAIN/'src/engine/num/value.rs']:bind(p)
    a=WORK/'aeneas';c=WORK/'charon'
    archive(os.environ['AENEAS_SRC'],selected['AENEAS_COMMIT'],selected['AENEAS_TREE'],a,'aeneas')
    archive(os.environ['CHARON_SRC'],selected['CHARON_COMMIT'],selected['CHARON_TREE'],c,'charon')
    # Verify every registered base-file digest before applying its registered patches.
    for label,dest in [('AENEAS',a),('CHARON',c)]:
        listing=re.search(label+r'_BASE_SHA256="([^"]+)"',s).group(1)
        for line in listing.splitlines():
            digest,path=line.split();p=dest/path
            if sha(p)!=digest:raise RuntimeError('STEP_BASE_DIGEST '+str(p))
            bind(p)
    patch(a,ROOT/'proof/patches/aeneas-array-repeat.patch','registered-aeneas')
    patch(c,ROOT/'proof/patches/charon-repeat-optional-trait.patch','registered-charon')
    link=a/'src/charon'
    if link.is_symlink():link.unlink()
    link.symlink_to(c,target_is_directory=True)
    # Existing OCaml cache is a private COW acceleration only; dune rechecks archived sources.
    cow(Path(os.environ['AENEAS_SRC'])/'src/_build',a/'src/_build','ocaml-cache-cow')
    lean=a/'backends/lean';run(['rm','-rf',lean],ROOT,'private-lean-remove')
    cow(MAIN/'target/aeneas-model/lean',lean,'lean-model-cow')
    packages=lean/'.lake/packages'
    if packages.is_symlink():packages.unlink()
    else:run(['rm','-rf',packages],ROOT,'private-package-remove')
    cow(MAIN/'proof/.lake/packages',packages,'lean-deps-cow')
    # Reject symlinked caches that could write back into a shared lane.
    for p in packages.rglob('*'):
        if p.is_symlink() and not p.resolve().is_relative_to(packages.resolve()):raise RuntimeError('STEP_SHARED_CACHE_SYMLINK '+str(p))
    base=lean/'Aeneas/Std/Core/Iter.lean'
    if sha(base)!='c59a9f6ba5a05e528a658be2fdc8e621c16bcd72374e88adec6c7119d009a92c':raise RuntimeError('STEP_MODEL_BASE_DRIFT')
    bind(base)
    run(['patch','-p1','--dry-run','-R','-i',ROOT/'proof/patches/aeneas-lean-model.patch'],lean,'registered-model-reverse-check')
    patch(lean,ROOT/'proof/native/step_compat.patch','private-step')
    bind(base)
    seed=MAIN/'target/native-step-model-cache'
    prepared=MAIN/'target/native-step-model-cache/source-identity.json'
    if not prepared.is_file():raise RuntimeError('STEP_PREPARED_CACHE_IDENTITY_MISSING')
    identity=json.loads(prepared.read_text());bind(prepared)
    if identity.get('exit')!=0 or identity.get('destination')!=str(seed) or identity.get('aeneas_commit')!=selected['AENEAS_COMMIT'] or identity.get('charon_commit')!=selected['CHARON_COMMIT']:raise RuntimeError('STEP_PREPARED_CACHE_IDENTITY_INVALID')
    if not (seed/'.lake/build/bin/extract').is_file():raise RuntimeError('STEP_NATIVE_CACHE_MISSING')
    source_manifest={}
    sources=list(lean.joinpath('Aeneas').rglob('*.lean'))+[lean/'Aeneas.lean',lean/'AeneasExtract.lean',lean/'lakefile.lean',lean/'lake-manifest.json',lean/'lean-toolchain']
    for current in sources:
        relative=current.relative_to(lean); previous=seed/relative
        if not previous.is_file() or sha(previous)!=sha(current):raise RuntimeError('STEP_NATIVE_CACHE_SOURCE_DRIFT '+str(relative))
        source_manifest[str(relative)]=sha(current)
    # Check the complete dependency source population before reusing its native caches.
    for current in packages.rglob('*.lean'):
        relative=current.relative_to(packages)
        if '.lake' in relative.parts or '.git' in relative.parts:continue
        previous=seed/'.lake/packages'/relative
        if not previous.is_file() or sha(previous)!=sha(current):raise RuntimeError('STEP_NATIVE_DEPENDENCY_DRIFT '+str(relative))
        source_manifest['packages/'+str(relative)]=sha(current)
    if source_manifest!=identity['source_manifest']:raise RuntimeError('STEP_PREPARED_CACHE_POPULATION_DRIFT')
    manifest_path=WORK/'native-cache-source-manifest.json'
    manifest_path.write_text(json.dumps(source_manifest,sort_keys=True));bind(manifest_path)
    bind(seed/'.lake/build/bin/extract')
    run(['rm','-rf',lean/'.lake/build',packages],ROOT,'private-native-cache-remove')
    cow(seed/'.lake/build',lean/'.lake/build','native-model-cache-cow')
    cow(seed/'.lake/packages',packages,'native-dependency-cache-cow')
    run(['lake','env','lean','--version'],lean,'lean-version')
    run(['git','rev-parse','HEAD'],MAIN/'proof/.lake/packages/mathlib','mathlib-version')
    run(['lake','build','Aeneas:leanArts'],lean,'model-lean-build')
    run(['lake','exe','extract'],lean,'generate-registry')
    bind(lean/'.lake/build/bin/extract')
    reg=a/'src/extract/ExtractBuiltinLean.ml';text=reg.read_text();bind(reg)
    if 'StepUsize.forward_overflowing' not in text or '"forward_overflowing"' not in text:raise RuntimeError('STEP_REGISTRY_FIELDS_MISSING')
    charon=MAIN/'target/aeneas/charon';installed=MAIN/'target/aeneas/aeneas'
    if sha(installed)!=selected['AENEAS_TRANSLATOR_SHA256']:raise RuntimeError('STEP_INSTALLED_TOOL_DRIFT observed='+sha(installed)+' pinned='+selected['AENEAS_TRANSLATOR_SHA256'])
    for p in [charon,MAIN/'target/aeneas/charon-driver',installed,lean/'lean-toolchain',lean/'lake-manifest.json',lean/'AeneasExtract.lean']:bind(p)
    for argv,label in [([charon,'version'],'charon-version'),([charon,'toolchain-version'],'rustc-version'),([charon,'toolchain-path'],'rustc-sysroot'),([installed,'-version'],'installed-version'),(['dune','--version'],'dune-version'),(['ocamlc','-version'],'ocaml-version')]:run(argv,ROOT,label)
    rustsys=Path(run([charon,'toolchain-path'],ROOT,'sysroot-source')[1].strip())
    bind(rustsys/'lib/rustlib/src/rust/library/core/src/iter/range.rs')
    for rel,digest in [('num/error.rs','f66e9b263eb9f9aab957801d08a4a8185d7d15e5945a9f491023986d19ca2636'),('convert/num.rs','cef1248e8e76d58077b34f622c51d07f25d78a1293783a6bd7ae371f639b84db')]:
        path=rustsys/'lib/rustlib/src/rust/library/core/src'/rel
        if sha(path)!=digest:raise RuntimeError('STEP_SYSROOT_NARROW_SOURCE_DRIFT '+rel)
        bind(path)
    rust=WORK/'rust';rust.mkdir();manifest=rust/'Cargo.toml'
    manifest.write_text('[package]\nname="apeiron"\nversion="0.1.0"\nedition="2021"\n[lib]\npath="'+str(MAIN/'src/lib.rs')+'"\n[target.\'cfg(target_os = "linux")\'.dependencies]\nlibc="0.2"\n[workspace]\n')
    run(['cargo','generate-lockfile','--offline'],rust,'lock');bind(manifest)
    llbc=WORK/'ExactDotSource.llbc'
    run([charon,'cargo','--sysroot','default','--preset','aeneas','--dest-file',llbc,'--start-from','crate::engine::kern::gemm::exact_dot','--include','core::convert::num::{impl core::convert::TryFrom<_> for _}::try_from','--include','core::num::error::TryFromIntError','--include','core::num::error::IntErrorKind','--','--manifest-path',manifest,'--locked','--offline','--lib','--target-dir',WORK/'rust-target'],ROOT,'fresh-charon')
    bind(llbc);check_source(llbc);validate_narrow(json.loads(llbc.read_text()))
    shape_controls(llbc)
    translate(installed,llbc,WORK/'baseline','installed-baseline',red=True)
    env={'AENEAS_VERSION':selected['AENEAS_COMMIT']+'-private-step-array-repeat'}
    run(['dune','build','main.exe','-j','1'],a/'src','private-ocaml-build',env=env)
    binary=a/'src/_build/default/main.exe';bind(binary)
    run([binary,'-version'],ROOT,'private-version')
    translate(binary,llbc,WORK/'positive','private-positive')
    # This checkpoint comes only from the fresh positive executable in this invocation.
    checkpoint=WORK/'positive-ocaml-build';pristine=WORK/'registry-pristine.ml'
    positive_binary=sha(binary);positive_sources=ocaml_sources(a,c);positive_llbc=sha(llbc)
    run(['cp','-a',reg,pristine],ROOT,'registry-checkpoint');bind(pristine)
    cow(a/'src/_build',checkpoint,'positive-build-checkpoint')
    checkpoint_identity=WORK/'positive-build-identity.json'
    checkpoint_identity.write_text(json.dumps(dict(binary_sha256=positive_binary,
        llbc_sha256=positive_llbc,registry_sha256=sha(pristine),registry_mtime_ns=pristine.stat().st_mtime_ns,sources=positive_sources,
        inputs={str(p):sha(p) for p in [ROOT/'proof/native/step_gate.py',ROOT/'proof/native/step_compat.patch',ROOT/'proof/native/StepRefinement.lean',lean/'Aeneas/Std/Core/Iter.lean',lean/'lean-toolchain',lean/'lake-manifest.json']}),sort_keys=True));bind(checkpoint_identity)
    generated=(WORK/'positive/ExactDotSource.lean').read_text()
    adapted=bridge(generated);bind(MAIN/'proof/native/NarrowSource.lean')
    (lean/'StepRefinement.lean').write_bytes((ROOT/'proof/native/StepRefinement.lean').read_bytes())
    (lean/'ExactDotSource.lean').write_text(adapted)
    source_compile(lean,'actual-full-source')
    proof_check(lean,'actual-full-proof')
    semantic_controls(lean,adapted)
    original=pristine.read_text()
    # The generated trait is one line: remove the pair, never the declaration.
    needle='("forward_overflowing", "forward_overflowing"); '
    lines=original.splitlines(keepends=True)
    matches=[i for i,line in enumerate(lines) if 'mk_trait_decl "core::iter::range::Step" "core.iter.range.Step"' in line]
    if len(matches)!=1 or lines[matches[0]].count(needle)!=1:raise RuntimeError('STEP_REGISTRY_MUTATION_TARGET')
    target=lines[matches[0]];changed=target.replace(needle,'',1)
    other=[('steps_between','steps_between'),('forward_checked','forward_checked'),('backward_checked','backward_checked'),('backward_overflowing','backward_overflowing')]
    for rust,lean_name in other:
        pair='("'+rust+'", "'+lean_name+'")'
        if target.count(pair)!=1 or changed.count(pair)!=1:raise RuntimeError('STEP_REGISTRY_OTHER_FIELD_DRIFT')
    parents='~parent_clauses:["cloneInst"; "partialOrdInst"]'
    if target.count(parents)!=1 or changed.count(parents)!=1:raise RuntimeError('STEP_REGISTRY_PARENT_DRIFT')
    mutated=original.replace(target,changed,1)
    if mutated!=original.replace(needle,'',1) or 'mk_trait_decl "core::iter::range::Step" "core.iter.range.Step"' not in mutated:raise RuntimeError('STEP_REGISTRY_DECLARATION_DRIFT')
    reg.write_text(mutated);bind(reg)
    try:
        run(['dune','build','main.exe','-j','1'],a/'src','mutant-ocaml-build',env=env);bind(binary)
        translate(binary,llbc,WORK/'mutant','registry-mutant',red=True)
    finally:
        # Always restore private source bytes and mtimes, including failed controls.
        run(['cp','-a',pristine,reg],ROOT,'registry-byte-restore');bind(reg)
        if reg.read_bytes()!=pristine.read_bytes() or reg.stat().st_mtime_ns!=pristine.stat().st_mtime_ns:raise RuntimeError('STEP_REGISTRY_RESTORE_DRIFT')
    if ocaml_sources(a,c)!=positive_sources:raise RuntimeError('STEP_RESTORED_TOOL_SOURCE_DRIFT')
    run(['rm','-rf',a/'src/_build'],ROOT,'private-mutant-build-remove')
    cow(checkpoint,a/'src/_build','positive-build-restore')
    run(['dune','build','main.exe','-j','1'],a/'src','restore-ocaml-build',env=env);bind(binary);bind(reg)
    if sha(binary)!=positive_binary:raise RuntimeError('STEP_RESTORED_BINARY_DRIFT')
    run([binary,'-version'],ROOT,'restored-private-version')
    if sha(llbc)!=positive_llbc:raise RuntimeError('STEP_RESTORED_LLBC_DRIFT')
    bind(llbc)
    translate(binary,llbc,WORK/'restored','private-restored')
    restored=(WORK/'restored/ExactDotSource.lean').read_text()
    if restored!=generated:raise RuntimeError('STEP_TRANSLATION_RESTORE_DRIFT')
    if (lean/'ExactDotSource.lean').read_text()!=adapted:raise RuntimeError('STEP_SOURCE_RESTORE_DRIFT')
    source_compile(lean,'final-full-source')
    proof_check(lean,'final-full-proof')
    print('STEP_COMPATIBILITY_OK',flush=True)

if __name__=='__main__':
    status=1
    try:main();status=0
    finally:
        OUT.mkdir(parents=True,exist_ok=True)
        name=WORK.name if WORK else 'step-no-work'
        (OUT/(name+'-receipt.json')).write_text(json.dumps(dict(exit=status,wall_seconds=time.time()-float(os.environ.get('STEP_START',time.time())),budget=540,deadline=DEADLINE,argv=sys.argv,stages=RECORDS,sha256=HASHES,artifacts=ARTIFACTS,work=str(WORK),memory='12G',swap=0,threads=1,trust='private Step wiring, real std narrowing and exhausted exact_dot tail through Aeneas model Result.ok; Rust Result.ok source refinement, whole-loop arithmetic/cost, GPU/codegen/hardware excluded'),indent=2)+'\n')
