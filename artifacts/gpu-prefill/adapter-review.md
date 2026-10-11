# Independent adapter review

Decision: **REJECT pending the bounded fixes below**. The numeric mapping and stated proof boundary
are useful design anchors, but the executable gate admits four material defects. Worker files
remain untouched. This review does not register a source-cost contract or authorize implementation.

Owner Sol; documentation exception in supplied Apeiron checkout. Candidate is Luna's three files
under `/tmp/worktree-agent-APEIRON-GPU-PREFILL-W1-T2/artifacts/gpu-prefill/`:
adapter-gate.py, adapter-identities.json, adapter-contract.md. Candidate base
`ad707f448e187697af0577dc2c5e5b610409f371`; Fold source
`d497161f32e0d366220f2065ec36e680bd9aed86`. Review budget 180 s; gate budget 120 s.

```text
SCOPE:      artifacts/gpu-prefill/adapter-review.md; independent verdict, no worker edits.
REGISTER:   planning/model checker review; source-cost registration and implementation excluded.
CORPUS:     candidate declares no completed adapter refinement; no fresh Lean check.
REFERENCE:  candidate adapter files; actual Fold simt_nn/batch.rs and simt/superblock.rs;
            actual Apeiron format/quant.rs and engine/chart/mod.rs.
GATE:       timeout 120s python3 <candidate>/adapter-gate.py --check --controls;
            exit 0, independently timed 0.029185 s; four extra countercontrols survive.
GRADE:      listed controls tested; numeric correspondence anchor/model; adapter proof open.
```

## Correct source distinctions

- Signed Q8 recoding `u=q+128`, `DM=(d,-128*d)` has the stated exact arithmetic identity
  `d*sum(u*x)-128*d*sum(x)=d*sum(q*x)`. Fold's row-tiled Q8 path uses signed XQ and unsigned W
  with `OpSUDot`, k-packed W and token-major XQ; the proposed code orientation is correct.
- f16 bias representability is a real restriction: d=3/8 gives exactly representable -48;
  finite source f16 d=512 requires bias -65536, outside finite f16. The latter must refuse.
  The review executed both conversions with Python's binary16 packing; the candidate gate
  exercises neither its f16 success nor its f16 refusal policy.
- Apeiron `QuantAct` represents `scale_fixed*code` in Q32.32. Fold's XS comes from the
  pre-quantization sum, not the sum of original values implied by an arbitrary adapter.
  Reconstruction must use real values `(scale_fixed*code)/2^32`; SX is `scale_fixed/2^32`.
  Keeping original unquantized XS would change the affine correction term.
- Float association is stated correctly as retained Fold behavior. Actual
  `simt/superblock.rs:335-340` emits `(d*dot)*sx + lo*XS`, then adds to the running f32.
  Apeiron `gemm.rs:311-329` accumulates checked i128 and rounds once. The real arithmetic
  identity does not prove these observations equal. The candidate explicitly keeps that open.
- The q8 offset panic boundary is correctly stated. `quant.rs:235-241` checks b*34,
  then performs ordinary o+34 addition before `need`. A total Result refusal theorem for every
  invalid offset cannot be inferred from the checked multiplication.
- Dense Fold resident prefill, hybrid decode recurrence and missing hybrid batched recurrence
  remain distinct. Session generations and checked capacities are proposed adapter behavior,
  not existing Fold pointer-owner behavior. No runtime continuation/reset acceptance follows.

## Surviving countercontrols

The review imported the actual candidate checker, deep-copied its actual JSON, changed one field,
then called `validate(mutant, check_sources=True)` and `model_witnesses()`.
All four returned successfully:

| mutation | observed result | obligation omitted by the gate |
|---|---|---|
| `sources=[]` | SURVIVES | exact source-set completeness; every digest check is skipped |
| `dm_address="DM[row*blocks+block]"` | SURVIVES | actual block-major DM address contract |
| `activation_gpu_encoding="signed i8 XQ; XS=sum(original unquantized input), SX=1"` | SURVIVES | signedness substring passes despite wrong XS and SX |
| `continuation="reuse foreign generation and stale prefix"` | SURVIVES | continuation policy field is not checked |

The five listed controls do reject their chosen changed strings/hashes. That is tested manifest
policy enforcement for those fields. It is not an attributable numeric/layout/state control:
`model_witnesses` hardcodes its own d/q/x and session dictionaries, without interpreting the
manifest fields being checked. Its empty and K refusal assertions (`not (0>0)`, `33%32!=0`)
do not call an adapter model. Source hashes do not repair that disconnected witness boundary.

## Bounded fixes: same three candidate files, same gate

1. Require exactly the ten expected unique source paths and their required symbol sets; reject
   missing, duplicate and unknown bindings. Add removal-of-one/all-source controls. Keep the
   checked source hashes; no broader source/refinement task is allocated here.
2. Encode DM and XS/SX addressing, Q32 units, activation preservation and continuation conditions
   as structured fields that `validate` checks. Make a small adapter *model* consume those fields
   and execute byte packing/transpose, SX/XS construction, checked sizes and session eligibility.
   Add two rows/two blocks/two tokens so wrong strides produce a different observed result;
   mutate DM stride, XS convention, SX scaling and stale generation/prefix. The existing fixed
   rational identity can remain as one algebra witness, labeled model.
3. Exercise finite/exact f16 DM conversion with d=3/8 success and d=512 rejection, NaN/Inf,
   zero scales, live/padded output extents and checked shape overflow. Require the mutated
   acceptance policy to fail a witness, then restore. These remain model controls, not GPU proof.
4. Clarify the reconstructed-XS operation: **do not run reconstructed QuantAct through Fold's
   existing quantize_at to rebuild all XQ/SX**. That source (`simt_nn/batch.rs:53-130`) computes
   absmax/127 scales again. A valid Apeiron code64, fixed scale2 reconstructs fixed value128;
   Fold's re-quantizer emits code127 and SX=128/(127*2^32), rather than retaining code64 and
   SX=2/2^32. Directly pack the existing signed codes and preserve their scale, computing only
   XS from reconstructed values in the specified order. Alternatively register re-quantization
   as changed observations; that is outside this adapter contract.

The repair stays within adapter-gate.py, adapter-identities.json and adapter-contract.md, one
gate and a fresh independent rerun. Candidate code remains a model: no production converter,
Rust extraction/refinement, emitted-module value theorem, GPU execution or performance acceptance
is added by these repairs. Actual buffer reclamation, constructor capacities, shared state and
driver/device behavior retain their own gates.

## Independent gate receipt

Exact command:

```text
timeout 120s python3 /tmp/worktree-agent-APEIRON-GPU-PREFILL-W1-T2/artifacts/gpu-prefill/adapter-gate.py --check --controls
```

Exit 0; wall elapsed 0.029185 s, measured around subprocess execution. Output:

```text
SOURCE_BOUND_CHECK:PASS (identities, symbols, contract, model witnesses)
CONTROL_RED:weight_signedness:CONTRACT_MISMATCH:weight_gpu_encoding
CONTROL_RESTORED:weight_signedness
CONTROL_RED:weight_stride:CONTRACT_MISMATCH:weight_stride_bytes
CONTROL_RESTORED:weight_stride
CONTROL_RED:padded_extent:CONTRACT_MISMATCH:padded_extent
CONTROL_RESTORED:padded_extent
CONTROL_RED:session_generation:CONTRACT_MISMATCH:session_identity
CONTROL_RESTORED:session_generation
CONTROL_RED:source_sha256:IDENTITY_DIGEST_MISMATCH:/tmp/worktree-agent-APEIRON-GPU-PREFILL-W1-T2/src/format/quant.rs
CONTROL_RESTORED:source_sha256
GRADE: source-bound anchor + model; no Rust extraction/refinement or GPU execution
```

| candidate artifact | sha256 at review |
|---|---|
| adapter-gate.py | `f336b63bdf723529317c7f319c257b787c3babc6a87dbe623fc524fb2f85a4cd` |
| adapter-identities.json | `b0da950f0f10fd00be9c770d8846ea65feb681b541fdc36989d8540c80fc8868` |
| adapter-contract.md | `4bdcee1608d63fdcf1c57935dc9925371250f0ef1898c35e76fb781b3ec46bc1` |

The independently run gate checked candidate source digests against the actual files. Important
read source identities include Apeiron quant.rs
`e574ea15520ed3c17dbfb90d514c2134a3883f681bbfba63b1eb200992f65335`, chart/mod.rs
`ae4a4a247f01b8435cb93860554d4b7d795b4c41d804b1d27abe614f23ad68c0`, Fold batch.rs
`d183f9cb8decb9ffccb37c91a7973c39003b33361babc8dc6f98615cfc618fd7`, superblock.rs
`e15e18bacfa02b5508dfefc959fc9bd7f2ef1593ff208d8de6e5e3c716745387`, and gpu.rs
`1e01838dca24f7751274249af671bb485109b80f8f0e7f9fa940e73744e9b17c`.

Next bounded task: repair those four checker/model gaps and re-review the same three files.
The earlier resource-census slice remains budget-failed (539/480 s); this independent adapter
review does not close it or inherit its acceptance.
