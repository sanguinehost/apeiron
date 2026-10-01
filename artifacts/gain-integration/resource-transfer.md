# Source-bound CPU/GPU resource census

Decision: the executing shared primitive is grouped quantized projection with explicit packing,
row/token tiles and state lifetimes. Reuse Fold's existing GPU graph; its typed kernel choices are
the first finite resource-selection domain. Preserve every float association. Counts below have
grade `model`; the checker and its manifest controls have grade `tested`. No inference, GPU,
benchmark, admission, Rust build or Lean command runs in this task.

Task `APEIRON-RESOURCE-W1-T1`, owner Sol, worktree `/tmp/worktree-agent-APEIRON-RESOURCE-W1-T1`,
branch `codex/resource-census-w1`, base `ad707f448e187697af0577dc2c5e5b610409f371`.
Owned files: resource-census.py, resource-census.json, resource-transfer.md in this directory.
Slice budget 480 s; task gate 60 s; no shared build lane. Lead owns integration and commits.

```text
SCOPE:      three census artifacts; actual-source identities, workload shapes, counts and controls.
REGISTER:   source-bound resource model and checker test; no executing-source cost proof or speed claim.
CORPUS:     apeiron_word.q8_0_codes_ok in proof/corpus/Numeric/DequantQ8_0.lean:
            b.val*34+34 <= raw.val.length implies the extracted accessor returns 32 signed codes.
REFERENCE:  /home/socol/Workspace/fold/modelfold/src/simt/superblock.rs;
            src/engine/{step/cols.rs,kern/gemm.rs,chart/ffn.rs,chart/pool.rs} read in this worktree.
GATE:       timeout 60s python3 artifacts/gain-integration/resource-census.py --check --controls;
            baseline exit 2: script absent. Slice budget 480 s.
GRADE:      model counts; tested checker/control; hardware counters UNKNOWN.
```

## Identities and actual workloads

The executable JSON binds ten CPU files at retained `04d206013f70f245dfdaf2619c0b1846d10b9922`
(read through `git show`) and their current digests, seven live Fold files at
`d497161f32e0d366220f2065ec36e680bd9aed86`, the actual integration source manifest, raw receipts
and archived ratios. Each source records whether current/retained/combined digests match.
Dirty integration and historical benchmark identities stay separate. Artifacts absent from the
task worktree are read from `/home/socol/Workspace/apeiron/artifacts/track-consolidation/`;
those absolute dependencies are explicit and checked, not embedded invented receipts.

| actual GGUF directory | prompt | trunk / hidden / FFN | CPU block widths |
|---|---|---|---|
| Qwen3.5-0.8B Q8_0 | 200 | 24 / 1024 / 3584 | 200 |
| Qwen3.5-0.8B Q8_0 | 512 | 24 / 1024 / 3584 | 256,256 |
| Qwen2.5-0.5B Q8_0 | 200 | 24 / 896 / 4864 | 200 |
| Qwen2.5-1.5B Q8_0 | 200 | 28 / 1536 / 8960 | 200 |

GGUF metadata/tensor directory bytes are freshly read and hashed; payload bytes are not freshly
authenticated. Historical model payload hashes stay labeled historical. Qwen3.5 block_count 25
includes one MTP block: loader.rs:620-630 subtracts nextn_predict_layers, leaving 24 trunk layers.
All actual projection tensor shapes come from the files, rather than a recalled architecture.
The script accounts grouped softmax Q/K/V, affine QKV/gate/alpha/beta, mixer output, fused
FFN gate/up and down. Fused gate/up materializes one SwiGLU output, not two projection outputs.

Four selected receipts carry five quiet rounds, hybrid pp200/pp512 and dense pp200, against
llama.cpp `0324696b8` with 8/16/24/32 thread candidates on Intel Core i9-14900KF.
Their JSON contains raw historical trace observations and exact comparator identities. They are
end-to-end ratios; neither the census nor those receipts establish causal before/after ablation.
Current CPU execution and actual selected GPU path are UNKNOWN: this task does not run inference.
An archived phase trace is historical execution evidence, not a new fast-branch witness.

## Derived CPU counts and residuals

Let B be one token block, K the projection width, A=K/32, G=ceil(B/8), and M the output rows.
The successful GEMM branch in kern/gemm.rs:131-200 allocates four packing buffers. Its loops write:

| count | derivation |
|---|---|
| packed word bytes | 4*G*A*8*8 |
| packed shift bytes | 8*G*A*8 |
| emin/spread bytes | 4*B + 4*G |
| packing reads | B*K code bytes and 16*B*A scale bytes (two scale scans) |
| packing dispatch | one pool run over G groups |
| grouped row dispatch | one run, sum ceil(M_part/rpt) tasks; rpt=16 for a group, 4 for a single M<=1024 |
| scalar MAC terms | B*K*sum(M_part); output bytes 8*B*sum(M_part), except fused SwiGLU's 8*B*M_ff |
| fused SwiGLU tasks | ceil(M_ff/16); two projections' products feed one output without two intermediates |
| quantization copies | B*(K+8*A), one column pool dispatch (chart/ffn.rs:148-210) |
| quantizer buffers | 2*B persistent + 2*B temporary code/scale buffers; outer vectors/pointers separate |

Grouping shares one activation quantization/pack across QKV or four affine input projections;
FFN gate/up shares a pack and row dispatch. This is the concrete common computation that can
explain transfer to dense models. It is a source-derived explanation, not measured attribution.

Literal CPU weight loads are not equal to distinct payload bytes. kern/gemm.rs:1263-1296 walks
row pairs, then pairs of eight-column groups; group2_x2 reads weight codes again per group pair.
For a full row pair this gives ceil(G/2) code passes. Odd rows, shift/exact fallback and wider
exponent spread change load counts; hardware DRAM bytes and final instruction counts are UNKNOWN.
Pool waiting overlaps other phase timers and is never added as a new work count.

Refusals/fallbacks: unsupported type/ISA, empty shape, K not block-aligned, K/32>512, short
payload, invalid QuantAct magnitude/scales and unsafe exponent spread route to reference work.
matmat.rs additionally checks length multiplication and preserves first-column refusal order.
GEMM None is a fallback, not a completed projection. Whole allocation counts, peak live storage,
critical-path time and actual branch populations stay UNKNOWN. Named pack/quant storage counts
are not whole-model peaks or allocator measurements.

## Derived GPU counts and state

For superblock.rs:55's per-32 register-tiled kernel, with P=ceil(B/bt)*bt,
I=local*ceil(M/(local*rt))*(P/bt), A=K/32 and cpb=32/carrier.pack():

| emitted logical count | exact source-loop expression |
|---|---|
| global load words | I*A*(rt*(1+cpb)+10*bt): DM+weight words, eight XQ words/token, XS+SX |
| integer dot4 ops | I*A*8*rt*bt |
| float multiply/add ops | I*A*3*rt*bt / I*A*2*rt*bt |
| live output stores | P*M; row loads clamp to M-1, stores retain row predicate |
| chart distinct bytes | 4*M*(K/pack+K/32) |
| projection-only upload | P*K+8*P*A; result readback 4*B*M; one queue drain |

These count executed instructions in the modeled emitted schedule, including padded token/row work,
not GPU cache traffic. Source-to-emitted-code/resource correspondence is OPEN. Existing tile
selection (`gpu.rs:3992`) chooses bt8/32 and rt by accumulator/slab/tail constraints; the script
uses its no-override Q8 local128 policy explicitly. Fold configuration/packing for these Apeiron
GGUFs has not executed. GPU projection examples are conditional interfaces, not coverage claims.

Dense GpuPrefillStack uses 11*layers+2 dispatches for all logits, 11*layers+3 for last logits
(replace the full head with gather+one-tile head), and one continuation drain/chunk. Upload is
4*B*hidden+16. Hidden/logit host-visible writes are 4*B*(hidden+vocab), or
4*B*hidden+4*head_bt*vocab. The latter includes the whole padded head tile, not just one row.
KV mirrors additionally write 8*layers*B*nkv*(hd+1) device-to-host bytes; the runner's b_down
counter omits these KV writes, so the census lists them separately. Host folding writes
2*layers*B*nkv*hd code bytes + 8*layers*B*nkv scale bytes.

Same state owner and sufficient dev_len avoid reseeding. A changed owner uploads the prefix;
an advanced CPU state uploads only the missing gap, 8*layers*nkv*(hd+1) bytes/position.
Extra drains depend on staging size, owner and prefix; UNKNOWN here. GpuPrefillStack's dense
constructor does not cover hybrid ragged QKV; hybrid GpuDecode already owns recurrence buffers
and reset, but batched recurrence coverage stays open. CPU fallback is not GPU coverage.
Resident buffer allocation/peak totals and cold loading counts require the actual Fold loader.

## Frozen held-out resource prediction

Previously unused within this four-workload census: Q8 M3072,K2048,B384,local128, bt8.
Change rt1 to rt4 in the existing register-tiled kernel: logical global load words fall exactly
67.4157303371%; dot and valid-store counts stay equal; packing bytes and dispatch count change zero.
Both row and token extents divide exactly. This predicts source-model load work only, no latency.
Falsifier: actual emitted per-binding loads differ, dot/store extents differ, or the adapter
introduces a repack/transfer. Hardware cache/register effects are outside this prediction.

## Immediate source-cost extraction

Select real src/format/quant.rs::q8_0_codes, called through #[path] in an isolated extraction.
The existing apeiron_word.q8_0_codes_ok statement in Numeric/DequantQ8_0.lean:141 is the functional
link. A cost lift over its actual loop should show 32 iterations/byte reads/signed casts/array
writes on success, zero loop work for checked-multiply/truncation refusal, with positive and
short-payload witnesses. Exclude or account panic from unchecked o+34 addition when multiplication
fits but addition overflows. This is a scalar accessor cost theorem, not an AVX-VNNI theorem.
At most three owned extraction/generated/refinement files; preserve the real production root.
Mutation: change the actual loop increment or code offset in a throwaway source copy and require
the source-bound cost/value gate to reject, restored green. Parent separately checks the real act_scale source; this Q8 accessor cost link remains unrun.
No source-cost proof is claimed by this census.

## Acceptance receipt

Baseline task gate: exit 2, script absent. Creation command --write --check --controls: exit 0,
elapsed_seconds=0.778174. The required final --check --controls is recorded in the handback.

```text
PASS baseline: 17 live/retained sources, 4 GGUF workload directories, packing/tile/graph counts match
PASS control wrong-source: rejected SOURCE_BINDING
PASS restore wrong-source
PASS control wrong-shape: rejected SHAPE_BINDING
PASS restore wrong-shape
PASS control omitted-packing: rejected PACKING_OR_COST
PASS restore omitted-packing
PASS held-out: exact logical load reduction 0.674157303; equal dots/stores
GRADE model; checker/control execution tested; no Rust/SPIR-V proof or hardware measurement
```

Controls mutate disposable in-memory census records; they establish fail-closed manifest/shape/cost
checking, not semantic rejecting controls on Rust/SPIR-V proofs. Formula review remains trusted;
actual extraction is the next gate. External trust includes Python, git/IO, GGUF directory parsing,
raw receipt provenance, compiler/SIMD, Vulkan runtime/driver/device and unreclaimed Fold handles.

Worker preparation exceeded its 480 s slice: 539 s, failed. That record remains failed.
Independent Luna verification accepts these artifacts at model plus tested checker scope, with
the held-out rt1/rt4 result a hypothetical scenario (current Fold policy selects rt8).
The lead copies only these three artifacts and reruns the unchanged 60 s gate on main.
Source-cost/GPU/runtime/epic acceptance is not discharged by that artifact acceptance.
