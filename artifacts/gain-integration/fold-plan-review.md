# Independent Fold GPU plan review

**Verdict: accept the plans for bounded source/contract work.** They reuse Fold's current typed SPIR-V kernels and resident runner, prioritize hybrid prefill, preserve float association, and keep retained GEMV numbers at kernel-only historical scope. They do not propose another backend.

The source references resolve at Fold `d497161f32e0d366220f2065ec36e680bd9aed86`: `GemvSpec`/`legalize` in `/home/socol/Workspace/fold/modelfold/src/simt_nn/spec.rs:316,472`; row/token tiled `gemv_dp4a_batch_rt_module` in `/home/socol/Workspace/fold/modelfold/src/simt/superblock.rs:55`; resident `GpuPrefillStack` in `/home/socol/Workspace/fold/bedrock/src/serve/gpu.rs:4573,4711`; DeltaNet decode kernels in `/home/socol/Workspace/fold/modelfold/src/simt_nn/deltanet.rs:48`. `GpuPrefillStack::new` expects fused, uniform dense QKV/body shapes (`gpu.rs:4738-4750`), so the plan correctly treats batched hybrid DeltaNet scheduling as additional work, with dense integration only validating shared machinery (`gpu-prefill.md:60-63`). The hybrid objective and pp200/pp512 gate are explicit (`gpu-prefill.md:36-40`).

Float arithmetic and evidence framing are sound. `superblock.rs:326-340` fixes the existing f32 scale/bias operation order; the plan keeps integer exactness separate from the float observation (`gpu-prefill.md:65-71`). Its historical RTX 5080 figures distinguish matched f16 (155.9/154.1 µs and 137.8/131.1 µs) from q4dp4a versus f16 (2.53×/2.47×), and explicitly say these are not whole-prefill timings (`gpu-prefill.md:16-22`). The standalone receipts are linked in [fold-gpu-receipts.md](/home/socol/Workspace/apeiron/artifacts/gain-integration/fold-gpu-receipts.md).

The first tasks are bounded: `APEIRON-GPU-PREFILL-W1-T2` is an adapter contract, identity manifest and checker, at most three paths; the resource epic begins with a three-artifact census (`gpu-prefill.md:86-90`, `resource-guided-evolution.md:78-88`). The adapter design is a prerequisite, not itself the source-cost proof. **Action before implementation dispatch:** keep the source/refinement/resource gate marked MISSING until that contract selects one actual production body, its constructor-reachable success/refusal domain, and a red source-cost control. This sequencing is already stated in both plans (`gpu-prefill.md:102-110`; `resource-guided-evolution.md:124-127`); no document correction is required.

## Read-only identity

Fold HEAD is `d497161f32e0d366220f2065ec36e680bd9aed86`. The seven Fold source SHA-256 values in `fold-gpu-reuse.md` matched live files (the successful gate below checked every entry). Plan SHA-256: `gpu-prefill.md` `216ca3e9c32a3ddde8cc39c4b6280ed55023558ad5ee5dcf5d96a25c4ff53dd8`; `resource-guided-evolution.md` `01bc58f6503762c80757b93d6c31641386c9a5a15296e57e95700f989c515547`.

## Gate and controls

From `/home/socol/Workspace/apeiron`, run the existing structure/Intent gate recorded in `artifacts/gain-integration/sol-epic-review.md`:

```sh
timeout 60s python3 -c 'from pathlib import Path; exec(Path("artifacts/gain-integration/sol-epic-review.md").read_text().split("```python\n", 1)[1].split("```", 1)[0])'
```

Then run this Fold reference/digest check. It verifies pinned HEAD, each digest embedded in the reuse manifest, live source hashes and required plan anchors; both throwaway mutations must reject and restore:

```sh
timeout 60s python3 - <<'PY'
from pathlib import Path
import hashlib, subprocess, tempfile
root=Path.cwd(); fold=Path('/home/socol/Workspace/fold')
docs=[root/'docs/epics/gpu-prefill.md',root/'docs/epics/resource-guided-evolution.md']
rev='d497161f32e0d366220f2065ec36e680bd9aed86'
manifest=(root/'artifacts/gain-integration/fold-gpu-reuse.md').read_text()
rows={'modelfold/src/simt/dp4a.rs':'ffe4f6c313492b84ecfc2c5d99c9fa8b2945f5bf081282e9b13b1374da50901e','modelfold/src/simt/superblock.rs':'e15e18bacfa02b5508dfefc959fc9bd7f2ef1593ff208d8de6e5e3c716745387','modelfold/src/simt_nn/prefill.rs':'0d5da9a491d5223d1f8f9cf6b8d02a64d23b373bb2cef74384845b09c3e54d75','modelfold/src/simt_nn/batch.rs':'d183f9cb8decb9ffccb37c91a7973c39003b33361babc8dc6f98615cfc618fd7','modelfold/src/simt_nn/deltanet.rs':'c72f1859e82a974339b208ccf6ae67455fa7be73b85f6845887c5d6b69906879','modelfold/src/simt_nn/spec.rs':'f54e9fe3590aa4c6c87d382a8a4c9b76a56376eda58d212e4f2dd725a2c445b0','bedrock/src/serve/gpu.rs':'1e01838dca24f7751274249af671bb485109b80f8f0e7f9fa940e73744e9b17c'}
assert subprocess.check_output(['git','-C',str(fold),'rev-parse','HEAD'],text=True).strip()==rev
for p,h in rows.items(): assert h in manifest and hashlib.sha256((fold/p).read_bytes()).hexdigest()==h,p
def check(ss):
 return all('## Intent' in s and rev in s and all(x in s for x in ('modelfold/src/simt_nn/spec.rs','modelfold/src/simt/superblock.rs','bedrock/src/serve/gpu.rs')) for s in ss)
base=[p.read_text() for p in docs]; assert check(base); print('baseline PASS; Fold revision and 7 source hashes PASS')
with tempfile.TemporaryDirectory() as d:
 ps=[Path(d)/p.name for p in docs]
 def put(ss):
  for p,s in zip(ps,ss): p.write_text(s)
 def get(): return [p.read_text() for p in ps]
 bad=base.copy(); bad[0]=bad[0].replace('modelfold/src/simt_nn/spec.rs','modelfold/src/simt_nn/spec_missing.rs',1); put(bad); assert not check(get()); print('wrong Fold reference REJECT')
 put(base); assert check(get()); print('reference RESTORE PASS')
 bad=base.copy(); bad[0]=bad[0].replace('## Intent','## Removed',1); put(bad); assert not check(get()); print('missing Intent REJECT')
 put(base); assert check(get()); print('Intent RESTORE PASS')
PY
```

Results: existing document gate exit `0` in `0.040 s`; missing Intent rejected with `DOC_SECTION_MISSING`, restored copy passed. Fold reference/digest gate exit `0` in `0.004 s`; wrong reference rejected, then restored; missing Intent rejected, then restored. Combined wall time is under `0.1 s`, within the 60-second limit. No builds, benchmarks, GPU runs or Fold writes occurred; the 180-second review window was respected.
