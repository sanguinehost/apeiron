# Bounded epic review

Decision: retain the CPU gains and their cross-model transfer as the starting result; accept these
two documents as planning drafts, with implementation dispatch closed. Review owner: Sol. Supplied
checkout exception; base `e33977aa72534ad56775110be99cf002ca96fa37`; no build lane.

```text
SCOPE: docs/epics/gpu-prefill.md, docs/epics/resource-guided-evolution.md, this review.
REGISTER: planning; no runtime/proof/registry, governing document or Intent changes.
CORPUS: Spirv.checkFragment_header: acceptance implies parseHeader success;
        Spirv.gpu_dp4a_word_refines_cpu_word: Refines (gpu_dp4a_word ...) (cpu_dp4a_word ...).
        Actual source read; no fresh Lean check or concrete-module theorem claimed.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Spirv/Check.lean,
           /home/socol/Workspace/apeiron/proof/corpus/Spirv/Dp4a.lean,
           /home/socol/Workspace/apeiron/proof/DispatchAdmission.lean.
GATE: timeout 60s document check below, cwd /home/socol/Workspace/apeiron.
      Negative control removes Intent in a throwaway document; exit 1 required.
GRADE: anchor/planning; historical measurements retained; fresh build/measurement unrun.
```

| finding | decision |
|---|---|
| Retained hybrid pp200/pp512 ratios 2.17x/2.45x; dense pp200 1.58x/1.77x | Explain transfer through shared source costs first; no new dense 2x gate |
| GPU model words meet at canonical `dp4a_gemv_spec` | Reuse the integer seam for one resident projection block |
| Header/scans/apply-shape components exist; full checker composition and actual module/CPU-body links are absent | Keep exact open obligations; build success alone cannot supply the missing theorem |
| `gpu-spv.md` contract table uses proved labels for explicit unstated targets; verification table/source call them open | Lead reconciliation before proof dispatch; this review does not change that authority |
| Served captured fixture includes five float epilogue opcodes | Syntax acceptance is not integer execution; preserve 37 executing/42 served distinction |
| Registry is empty; `Admission.preserves` fixes exact Output equality and `improves` fixes strict Nat cost | Gate-first source/refinement/resource tasks only; changed model/harness observations require separate contract design |
| Current integrated candidate differs from measured 04d2060 | Keep source identities distinct; no inherited current throughput claim |

Next bounded task: `APEIRON-RESOURCE-W1-T1`, Luna owns the three census artifacts named in
the resource epic; Sol supplies independent review and the lead integrates.
Construct its missing check with source/shape/packing rejection and restoration, then inventory
retained shared paths and freeze a held-out resource prediction. No inference body is changed.
GPU prerequisite: existing `timeout 540s cargo test --test gpu_spv_check --locked` after the target
lane is released; record syntax/refusal scope and exact residual before allocating a kernel task.

The check below is document structure/path evidence only. It does not audit prose semantics or
establish theorem acceptance, source refinement, cost prediction, GPU throughput, or epic closure.

```python
import hashlib, json, subprocess, sys, tempfile, time
from pathlib import Path
started = time.monotonic()
root = Path('/home/socol/Workspace/apeiron')
paths = [root / 'docs/epics/gpu-prefill.md', root / 'docs/epics/resource-guided-evolution.md']
required = ['## Intent', '## Specification', '## Plan', '### Exit obligations']
def check(path):
    text = path.read_text()
    for section in required:
        if section not in text:
            raise ValueError(f'DOC_SECTION_MISSING: {path.name}: {section}')
    if 'MISSING' not in text or '540' not in text:
        raise ValueError(f'DOC_GATE_OR_BUDGET_MISSING: {path.name}')
for path in paths:
    check(path)
    text = path.read_text()
    assert 'Gemini' not in text, path
    assert 'Luna' in text and 'Sol' in text, path
    assert 'direct native dispatch is no bypass' in text, path
for rel in ['AGENTS.md', 'docs/direction.md', 'docs/kernel-boundary.md',
            'docs/plan.md', 'proof/README.md', 'README.md',
            'docs/epics/gpu-spv.md', 'proof/corpus/Spirv/Check.lean',
            'proof/corpus/Spirv/Dp4a.lean', 'proof/DispatchAdmission.lean',
            'tools/dispatch/contracts.json', 'src/spirv/check.rs',
            'tests/gpu_spv_check.rs', 'artifacts/track-consolidation/archived-ratios.json',
            'artifacts/track-consolidation/source-manifest.json']:
    assert (root / rel).is_file(), rel
assert 'theorem checkFragment_header' in (root / 'proof/corpus/Spirv/Check.lean').read_text()
assert 'theorem gpu_dp4a_word_refines_cpu_word' in (root / 'proof/corpus/Spirv/Dp4a.lean').read_text()
assert json.loads((root / 'tools/dispatch/contracts.json').read_text())['contracts'] == {}
with tempfile.TemporaryDirectory(prefix='sol-epic-doc-control-') as directory:
    scratch = Path(directory) / 'gpu-prefill.md'
    source = paths[0].read_text()
    scratch.write_text(source.replace('## Intent', '## Removed', 1))
    verifier = Path(directory) / 'check.py'
    gate_text = (root / 'artifacts/gain-integration/sol-epic-review.md').read_text()
    fence = chr(96) * 3
    gate_code = gate_text.split(fence + 'python\n', 1)[1].split(fence, 1)[0]
    checker_body = gate_code.split('def check(path):\n', 1)[1].split('\nfor path in paths:', 1)[0]
    verifier.write_text('import sys\nfrom pathlib import Path\n' +
        'required=' + repr(required) + '\ndef check(path):\n' + checker_body +
        '\ncheck(Path(sys.argv[1]))\n')
    red = subprocess.run([sys.executable, str(verifier), str(scratch)], capture_output=True, text=True, timeout=5)
    assert red.returncode == 1 and 'DOC_SECTION_MISSING' in red.stderr, red
    print(f'CONTROL_RED exit={red.returncode}: {red.stderr.strip().splitlines()[-1]}')
    scratch.write_text(source)
    restored = subprocess.run([sys.executable, str(verifier), str(scratch)], capture_output=True, text=True, timeout=5)
    assert restored.returncode == 0
    check(scratch)
    print(f'CONTROL_RESTORED exit={restored.returncode}')
for path in paths:
    print(path.relative_to(root), hashlib.sha256(path.read_bytes()).hexdigest())
print(f'DOC_CHECK_OK elapsed_seconds={time.monotonic()-started:.3f}')
```

Gate command (run at repository root):

```sh
timeout 60s python3 -c 'from pathlib import Path; exec(Path("artifacts/gain-integration/sol-epic-review.md").read_text().split("```python\n", 1)[1].split("```", 1)[0])'
```

Acceptance: document gate exit 0 in 0.050 s; missing-Intent copy exit 1 with
`DOC_SECTION_MISSING`; restored copy exit 0. Planning evidence only.
No builds, runtime mutations, Git mutations, commits or pushes.

Worker ownership follows the explicit user override: Luna bounded source/refinement/resource
worker, Sol independent reviewer, lead integration. No AGENTS.md edit or admission bypass.

Accepted draft sha256:

- `docs/epics/gpu-prefill.md`: `54ec684d24468a1001e0c64eb576dac1a9cabadedbb9212199503e87df7640cf`
- `docs/epics/resource-guided-evolution.md`: `596cf03efdbe8b39bd942921540044a64dd150fc9fc0221bb3ce5506b46399d8`
