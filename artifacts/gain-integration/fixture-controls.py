from pathlib import Path
import subprocess,os,json,time,tarfile,io,hashlib
start=time.monotonic();budget=150
root=Path('/tmp/apeiron-fixture-controls');out=Path('/home/socol/Workspace/apeiron/artifacts/gain-integration/fixture-controls');out.mkdir(parents=True,exist_ok=True);root.mkdir(exist_ok=True)
data=subprocess.check_output(['git','archive','04d206013f70f245dfdaf2619c0b1846d10b9922'],cwd='/home/socol/Workspace/apeiron')
with tarfile.open(fileobj=io.BytesIO(data)) as t:t.extractall(root,filter='data')
source=Path('/tmp/worktree-agent-APEIRON-FINALISE-W1-T1/src/engine/tests.rs').read_bytes();p=root/'src/engine/tests.rs';p.write_bytes(source)
env=os.environ.copy();env['CARGO_TARGET_DIR']='/home/socol/oc-tmp/wt/target-w6-integ';results=[]
def run(name,carrier,test,expected):
 env['APEIRON_F32_CARRIER']=carrier;argv=['cargo','test','--lib','--locked','--offline','engine::tests::'+test,'--','--exact']
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(argv,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,budget-(time.monotonic()-start)))
 log=(out/(name+'.log')).read_text();ok=(r.returncode==expected and ('assertion `left == right` failed' in log if expected else '1 passed; 0 failed' in log))
 results.append({'name':name,'exit':r.returncode,'attributable':ok});assert ok,name
run('baseline','1','attn_head_distinct_scores_pin_the_scale',0)
s=source.decode();start_ix=s.index('fn attn_head_distinct_scores_pin_the_scale()');end_ix=s.index('\n#[test]',start_ix);b=s[start_ix:end_ix];assert b.count('Carrier::Q32')==1
p.write_text(s[:start_ix]+b.replace('Carrier::Q32','Carrier::F32')+s[end_ix:]);run('carrier-red','1','attn_head_distinct_scores_pin_the_scale',101)
p.write_bytes(source);run('carrier-restored','1','attn_head_distinct_scores_pin_the_scale',0)
assert s.count('0x483c_6d3e_ce96_8247')==1;p.write_text(s.replace('0x483c_6d3e_ce96_8247','0x483c_6d3e_ce96_8246'));run('golden-red','0','engine_step_greedy_decode_is_deterministic',101)
p.write_bytes(source);run('golden-restored','0','engine_step_greedy_decode_is_deterministic',0)
assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(source).digest()
receipt={'budget':budget,'seconds':time.monotonic()-start,'source_sha256':hashlib.sha256(source).hexdigest(),'results':results,'restore_byte_identical':True};(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
