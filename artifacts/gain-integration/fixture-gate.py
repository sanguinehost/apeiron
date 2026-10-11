import os,subprocess,time,json,sys
from pathlib import Path
root=Path(sys.argv[1]); out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
names=['attn_head_distinct_scores_pin_the_scale','attn_saturates_out_of_window_exp_and_refuses_dot_overflow','attn_fused_den_num_are_independent_of_tile_count','attn_fused_refuses_zero_sum_and_zero_tile','engine_step_greedy_decode_is_deterministic']
start=time.monotonic(); results=[]
for carrier in ('0','1'):
 for name in names:
  env=os.environ.copy();env['APEIRON_F32_CARRIER']=carrier;env['CARGO_TARGET_DIR']='/home/socol/oc-tmp/wt/target-w6-integ'
  argv=['cargo','test','--lib','--locked','--offline','engine::tests::'+name,'--','--exact']
  with (out/(name+'-'+carrier+'.log')).open('w') as f:
   r=subprocess.run(argv,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,150-(time.monotonic()-start)))
  results.append({'carrier':carrier,'test':name,'exit':r.returncode})
receipt={'seconds':time.monotonic()-start,'budget':150,'results':results}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(any(r['exit'] for r in results))
