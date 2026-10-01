#!/usr/bin/env python3
"""Finite actual-loop trace: one bounded private kernel gate and terminal-branch control.
SCOPE: LoopTrace.lean, loop_trace_gate.py. REGISTER: model/count infrastructure only.
CORPUS: pinned Std.Primitives loop; WP loop.spec_decr_nat has no visit count.
REFERENCE: target/aeneas-model/lean/Aeneas/Std/Primitives.lean.
GATE: --controls; 540s including lane and descendants;12G/no swap/one Lean thread.
GRADE: generic model theorem only; Rust/source cost, optimization and GPU excluded.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, signal, subprocess, sys, tempfile, time
ROOT=Path(__file__).resolve().parents[2]
MAIN=Path('/home/socol/Workspace/apeiron')
OUT=Path('/tmp/opencode/native-loop-trace')
if __name__=='__main__' and os.environ.get('LOOP_TRACE_IN_LANE')!='1':
    now=time.time()
    os.execvp('timeout',['timeout','--kill-after=5','540s','flock','-w','530','/tmp/opencode/apeiron-lane.lock',
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0','env',
        'LEAN_NUM_THREADS=1','LOOP_TRACE_IN_LANE=1','LOOP_TRACE_START='+str(now),'LOOP_TRACE_DEADLINE='+str(now+532),
        sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
DEADLINE=float(os.environ.get('LOOP_TRACE_DEADLINE',time.time()+532))
STAGES=[];HASHES={};ARTIFACTS=[];WORK=None
PRIMITIVES='702946866c83fa0bb8b9d2a248c14164d38aa15e1eebd67caf4f39b0076d0ff8'
META_IDENTITY='d525325cd73446451381d27610dbaa5f1095c6afd4ae8d077dc24ced985ced5c'

def digest(path):
    """O(B) input bytes/O(1) streaming scratch."""
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def bind(path):
    value=digest(path);HASHES[str(path)]=value
    ARTIFACTS.append(dict(path=str(path),sha256=value,stage=STAGES[-1]['stage'] if STAGES else 'input'))

def run(argv,cwd,label,expected=0):
    """O(L) output bytes; every descendant consumes the common invocation deadline."""
    left=DEADLINE-time.time()-3
    if left<=0:raise RuntimeError('LOOP_TRACE_BUDGET_EXHAUSTED')
    log=OUT/(WORK.name+'-'+label+'.log');begin=time.monotonic();argv=list(map(str,argv))
    with log.open('wb') as f:
        p=subprocess.Popen(argv,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,
            env={**os.environ,'LEAN_NUM_THREADS':'1'},start_new_session=True)
        interruption=None
        try:p.wait(timeout=left)
        except BaseException as e:
            os.killpg(p.pid,signal.SIGKILL);p.wait();interruption=e
    text=log.read_text(errors='replace')
    STAGES.append(dict(stage=label,exit=p.returncode,seconds=time.monotonic()-begin,argv=argv,cwd=str(cwd),log=str(log)))
    print('LOOP_TRACE_STAGE',label,'exit='+str(p.returncode),f'seconds={time.monotonic()-begin:.3f}',str(log),flush=True)
    if interruption is not None:raise interruption
    if expected is not None and p.returncode!=expected:raise RuntimeError(label+'_FAILED '+str(p.returncode)+'\n'+text[-7000:])
    return p.returncode,text

def source_manifest(base):
    """O(B+F) model/dependency source bytes/files, O(F) digests; no artifact is acceptance."""
    manifest={}
    for p in list((base/'Aeneas').rglob('*.lean'))+list((base/'AeneasMeta').rglob('*.lean'))+[base/'AeneasMeta.lean',base/'Aeneas.lean',base/'AeneasExtract.lean',base/'lakefile.lean',base/'lake-manifest.json',base/'lean-toolchain']:
        manifest[str(p.relative_to(base))]=digest(p)
    for p in (base/'.lake/packages').rglob('*.lean'):
        relative=p.relative_to(base/'.lake/packages')
        if '.lake' in relative.parts or '.git' in relative.parts:continue
        manifest['packages/'+str(relative)]=digest(p)
    return manifest

def fresh_meta(model):
    """Recompile the actual imported helper and its native elaborator in the private copy."""
    paths=list((model/'.lake/build/lib/lean/AeneasMeta').glob('BvEnumToBitVec.*'))
    paths+=list((model/'.lake/build/ir/AeneasMeta').glob('BvEnumToBitVec.*'))
    paths+=list((model/'.lake/build/lib/lean').glob('aeneas_AeneasMeta_BvEnumToBitVec.so*'))
    for p in paths:
        if p.is_file():p.unlink()
    run(['lake','build','AeneasMeta.BvEnumToBitVec:leanArts','AeneasMeta.BvEnumToBitVec:dynlib'],model,'positive-meta-helper')
    for relative in ['.lake/build/lib/lean/AeneasMeta/BvEnumToBitVec.olean',
                     '.lake/build/lib/lean/aeneas_AeneasMeta_BvEnumToBitVec.so',
                     '.lake/build/ir/AeneasMeta/BvEnumToBitVec.c',
                     '.lake/build/ir/AeneasMeta/BvEnumToBitVec.c.o.export']:
        obj=model/relative
        if not obj.is_file():raise RuntimeError('LOOP_TRACE_META_OBJECT_MISSING '+str(obj))
        bind(obj)

def fresh_primitives(model,label):
    """Remove only this private module's build products so its actual source recompiles."""
    for p in (model/'.lake/build/lib/lean/Aeneas/Std').glob('Primitives.*'):
        if p.is_file():p.unlink()
    run(['lake','build','Aeneas.Std.Primitives:leanArts'],model,label)
    obj=model/'.lake/build/lib/lean/Aeneas/Std/Primitives.olean'
    if not obj.is_file():raise RuntimeError('LOOP_TRACE_PRIMITIVES_OBJECT_MISSING')
    bind(obj)

def proof_check(model,label,red=False):
    code,text=run(['lake','build','NativeLoopTrace:leanArts'],model,label,expected=None if red else 0)
    if red:
        source=(model/'LoopTrace.lean').read_text();start=source[:source.index('theorem projection ')].count('\n')+1
        end=source[:source.index('theorem deterministic ')].count('\n')+1
        errors=[(int(line),message) for line,message in re.findall(r'error: LoopTrace[.]lean:([0-9]+):[0-9]+: (.+)',text)]
        semantic=any(start<=line<end and ('unsolved goals' in message or 'Tactic' in message) for line,message in errors)
        if code!=1 or not semantic or 'LoopTrace.projection: forbidden dependencies [sorryAx]' not in text:raise RuntimeError('LOOP_TRACE_CONTROL_UNATTRIBUTED\n'+text[-7000:])
    elif 'LOOP_TRACE_KERNEL_OK' not in text:raise RuntimeError('LOOP_TRACE_AUDIT_MARKER_MISSING')

def main():
    global WORK
    parser=argparse.ArgumentParser();parser.add_argument('--controls',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(ROOT/'target').mkdir(exist_ok=True)
    WORK=Path(tempfile.mkdtemp(prefix='loop-trace-',dir=ROOT/'target'));print('LOOP_TRACE_WORK',WORK,flush=True)
    proof=ROOT/'proof/native/LoopTrace.lean'
    if not proof.is_file():raise RuntimeError('LOOP_TRACE_SOURCE_MISSING '+str(proof))
    if not args.controls:raise RuntimeError('LOOP_TRACE_CONTROLS_REQUIRED')
    text=proof.read_text()
    if re.search(r'\b(sorry|admit|axiom|native_decide)\b|set_option\s+(maxHeartbeats|maxRecDepth)',text):raise RuntimeError('LOOP_TRACE_PROOF_POLICY')
    for p in [proof,ROOT/'proof/native/loop_trace_gate.py',ROOT/'proof/patches/PINS']:bind(p)
    seed=MAIN/'target/native-step-model-cache';identity=seed/'source-identity.json'
    if not identity.is_file():raise RuntimeError('LOOP_TRACE_NAMED_CACHE_MISSING')
    d=json.loads(identity.read_text());bind(identity)
    if d.get('exit')!=0 or d.get('destination')!=str(seed) or d.get('aeneas_commit')!='505b6ca35217e7be5c96c3e2f8045edfbdf47291' or d.get('charon_commit')!='b104e24fea7d721b71e6c39fd70f26ff20bc0980':raise RuntimeError('LOOP_TRACE_CACHE_IDENTITY')
    supplement=seed/'meta-source-identity.json'
    if not supplement.is_file() or digest(supplement)!=META_IDENTITY:raise RuntimeError('LOOP_TRACE_META_PREPARED_IDENTITY')
    meta=json.loads(supplement.read_text());bind(supplement)
    if (meta.get('exit')!=0 or meta.get('destination')!=str(seed) or
        meta.get('aeneas_commit')!=d['aeneas_commit'] or
        meta.get('aeneas_tree')!='af23d5d9d866050c2158700cedab07501443bcdd' or
        meta.get('authority')!='immutable pinned git blobs; equal main and named-cache bytes' or
        meta.get('population')!=18 or meta.get('restored') is not True):raise RuntimeError('LOOP_TRACE_META_PREPARATION_BINDING')
    expected_meta=meta['source_manifest'];expected_original=d['source_manifest']
    if (len(expected_meta)!=18 or len(expected_original)!=9574 or
        set(expected_meta)&set(expected_original) or
        any(k!='AeneasMeta.lean' and not k.startswith('AeneasMeta/') for k in expected_meta)):
        raise RuntimeError('LOOP_TRACE_META_SOURCE_POPULATION')
    for relative,sha in expected_meta.items():
        for base in [seed,MAIN/'target/aeneas-model/lean']:
            p=base/relative
            if digest(p)!=sha:raise RuntimeError('LOOP_TRACE_META_AUTHORITATIVE_SOURCE_DRIFT '+str(p))
            bind(p)
    actual=source_manifest(seed)
    expected={**expected_original,**expected_meta}
    if len(actual)!=9592 or actual!=expected:raise RuntimeError('LOOP_TRACE_CACHE_SOURCE_POPULATION')
    pins=(ROOT/'proof/patches/PINS').read_text()
    for key,value in [('AENEAS_COMMIT',d['aeneas_commit']),('CHARON_COMMIT',d['charon_commit'])]:
        if not re.search(r'^'+key+'='+value+'$',pins,re.M):raise RuntimeError('LOOP_TRACE_PIN_DRIFT')
    relative=Path('Aeneas/Std/Primitives.lean')
    for p in [seed/relative,MAIN/'target/aeneas-model/lean'/relative]:
        if digest(p)!=PRIMITIVES:raise RuntimeError('LOOP_TRACE_ACTUAL_MODEL_DRIFT')
        bind(p)
    manifest_path=WORK/'source-manifest.json';manifest_path.write_text(json.dumps(actual,sort_keys=True));bind(manifest_path)
    model=WORK/'model'
    run(['cp','-a','--reflink=always',seed,model],ROOT,'private-model-cow')
    for p in model.rglob('*'):
        if p.is_symlink() and not p.resolve().is_relative_to(model.resolve()):raise RuntimeError('LOOP_TRACE_SHARED_CACHE_SYMLINK '+str(p))
    if source_manifest(model)!=actual:raise RuntimeError('LOOP_TRACE_PRIVATE_SOURCE_DRIFT')
    config=model/'lakefile.lean';config.write_text(config.read_text()+'\nlean_lib NativeLoopTrace where\n  roots := #[`LoopTrace]\n')
    bind(config)
    private_expected={**actual,'lakefile.lean':digest(config)}
    for p in [model/'lean-toolchain',model/'lake-manifest.json']:bind(p)
    (model/'LoopTrace.lean').write_bytes(proof.read_bytes());bind(model/'LoopTrace.lean')
    run(['lake','env','lean','--version'],model,'lean-version')
    run(['git','rev-parse','HEAD'],model/'.lake/packages/mathlib','mathlib-version')
    primitives=model/relative;original=primitives.read_bytes();bind(primitives)
    fresh_meta(model)
    if source_manifest(model)!=private_expected:raise RuntimeError('LOOP_TRACE_PRIVATE_POPULATION_DRIFT')
    fresh_primitives(model,'positive-primitives')
    proof_check(model,'positive-proof')
    needle=b'| ControlFlow.done x => ok x'
    if original.count(needle)!=1:raise RuntimeError('LOOP_TRACE_ACTUAL_DONE_ANCHOR')
    mutated=original.replace(needle,b'| ControlFlow.done x => fail .panic',1)
    primitives.write_bytes(mutated);bind(primitives)
    (WORK/'Primitives-mutant.lean').write_bytes(mutated);bind(WORK/'Primitives-mutant.lean')
    try:
        fresh_primitives(model,'mutant-primitives')
        proof_check(model,'mutant-proof',red=True)
    finally:
        primitives.write_bytes(original)
        if primitives.read_bytes()!=original or digest(primitives)!=PRIMITIVES:raise RuntimeError('LOOP_TRACE_RESTORE_DRIFT')
        bind(primitives)
    fresh_primitives(model,'restored-primitives')
    proof_check(model,'restored-proof')
    if source_manifest(model)!=private_expected:raise RuntimeError('LOOP_TRACE_FINAL_PRIVATE_POPULATION_DRIFT')
    for relative,sha in expected_meta.items():
        if digest(model/relative)!=sha:raise RuntimeError('LOOP_TRACE_PRIVATE_META_SOURCE_DRIFT')
        bind(model/relative)
    print('LOOP_TRACE_GATE_OK',flush=True)

if __name__=='__main__':
    status=1
    try:main();status=0
    finally:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/((WORK.name if WORK else 'loop-trace-no-work')+'-receipt.json')).write_text(json.dumps(dict(exit=status,wall_seconds=time.time()-float(os.environ.get('LOOP_TRACE_START',time.time())),budget=540,deadline=DEADLINE,argv=sys.argv,stages=STAGES,sha256=HASHES,artifacts=ARTIFACTS,work=str(WORK),memory='12G',swap=0,threads=1,trust='generic finite actual-loop model only; divergence, Rust/source cost/optimization/GPU/codegen/hardware excluded'),indent=2)+'\n')
