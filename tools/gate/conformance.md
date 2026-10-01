# Per-module gate script conformance

Machine twin: [`conformance.json`](conformance.json), one row per script matching
`proof/resource/*_gate.py`. Every field is read out of the script or out of the file the
script names. This record is a **binding of identities**, not evidence: running it
executes no gate and establishes nothing about the correctness of any theorem.

| fact | value |
|---|---|
| scripts on disk | 36 |
| tracked / untracked at HEAD | 31 / 5 |
| seeded mutation controls | 39 |
| generated at HEAD | `360b5ff1d7c1aa50d94d7c4a7f67844b938296ee` |

## Field reading order

1. targets.audit_marker: the literal string the script requires in the Lean build output.
2. targets.audited_headlines: the declarations that marker certifies, read from the resource file the script names.
3. targets.control_targets: headline names the script requires its own control to turn red.
4. mutation[].occurrences: the count the script itself asserts for its needle.

`mutation` is a list because `Gguf`, `Q6KCodes` and `RowDotAdmission` seed more than one
defect. `mutation[].occurrences` is the count the script itself asserts for its needle,
not a claim of this record.

## Lanes

| lock | scripts |
|---|---|
| `/tmp/opencode/apeiron-lane.lock` | 5 |
| `/tmp/opencode/apeiron-lane-1.lock` | 11 |
| `/tmp/opencode/apeiron-lane-2.lock` | 10 |
| `/tmp/opencode/apeiron-lane-3.lock` | 10 |

Every script self-bounds its whole invocation with an outer `timeout`, an inner
`timeout` under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`,
`LEAN_NUM_THREADS=1`, and a `flock` on its lane. Recorded as `budget_s`, `inner_s` and
`flock_wait_s`.

## Rows

| id | tracked | sha256 | green_marker | mutation file | controls | headlines | lane | budget/inner/flock |
|---|---|---|---|---|---|---|---|---|
| [`ApplyPending`](../../proof/resource/ApplyPending_gate.py) | yes | `625669d29536` | `APPLY_PENDING_SOURCE_RESOURCE_OK` | `src/table.rs` | 1 | 31 | `apeiron-lane-3.lock` | 540/525/530 |
| [`Argmax`](../../proof/resource/Argmax_gate.py) | yes | `0fe47e3665aa` | `ARGMAX_SOURCE_RESOURCE_OK` | `src/engine/sampler.rs` | 1 | 36 | `apeiron-lane-3.lock` | 540/525/530 |
| [`Capacity`](../../proof/resource/Capacity_gate.py) | yes | `2dfb8a810155` | `CAPACITY_SOURCE_RESOURCE_OK` | `src/table/resources.rs` | 1 | 19 | `apeiron-lane-3.lock` | 540/525/530 |
| [`CheckFragment`](../../proof/resource/CheckFragment_gate.py) | yes | `abcffb9b444e` | `CHECK_FRAGMENT_SOURCE_RESOURCE_OK` | `src/spirv/check.rs` | 1 | 19 | `apeiron-lane-3.lock` | 540/525/530 |
| [`CoordFinish`](../../proof/resource/CoordFinish_gate.py) | yes | `727d096bb9e4` | `COORD_FINISH_SOURCE_RESOURCE_OK` | `src/coord.rs` | 1 | 17 | `apeiron-lane-1.lock` | 540/525/530 |
| [`CoordId`](../../proof/resource/CoordId_gate.py) | yes | `aca9adb45681` | `COORD_ID_SOURCE_RESOURCE_OK` | `src/coord.rs` | 1 | 10 | `apeiron-lane-2.lock` | 540/520/525 |
| [`CoordPlace`](../../proof/resource/CoordPlace_gate.py) | yes | `dc307d09efc4` | `COORD_PLACE_SOURCE_RESOURCE_OK` | `src/coord.rs` | 1 | 39 | `apeiron-lane-3.lock` | 540/525/530 |
| [`CoordSpread`](../../proof/resource/CoordSpread_gate.py) | yes | `bb4ffebe0114` | `COORD_SPREAD_SOURCE_RESOURCE_OK` | `src/coord.rs` | 1 | 31 | `apeiron-lane-1.lock` | 540/525/530 |
| [`CoordUnspread`](../../proof/resource/CoordUnspread_gate.py) | yes | `4d20edc624c6` | `COORD_UNSPREAD_SOURCE_RESOURCE_OK` | `src/coord.rs` | 1 | 27 | `apeiron-lane-2.lock` | 540/525/530 |
| [`F16Split`](../../proof/resource/F16Split_gate.py) | yes | `fad15e1e7dec` | `F16_SPLIT_SOURCE_RESOURCE_OK` | `src/engine/kern/pow2.rs` | 1 | 21 | `apeiron-lane-2.lock` | 540/525/530 |
| [`FixedExp`](../../proof/resource/FixedExp_gate.py) | yes | `7740a947986f` | `FIXED_EXP_SOURCE_RESOURCE_OK` | `src/numeric/fixed_exp.rs` | 1 | 14 | `apeiron-lane-3.lock` | 540/525/530 |
| [`FixedRsqrt`](../../proof/resource/FixedRsqrt_gate.py) | yes | `196aa10a924f` | `FIXED_RSQRT_SOURCE_RESOURCE_OK` | `src/numeric/fixed_rsqrt.rs` | 1 | 9 | `apeiron-lane-1.lock` | 540/525/530 |
| [`FixedTrig`](../../proof/resource/FixedTrig_gate.py) | yes | `3d2bb299b778` | `FIXED_TRIG_SOURCE_RESOURCE_OK` | `src/numeric/fixed_trig.rs` | 1 | 20 | `apeiron-lane-1.lock` | 540/525/530 |
| [`Fwht`](../../proof/resource/Fwht_gate.py) | yes | `05e2536a1578` | `FWHT_SOURCE_RESOURCE_OK` | `src/engine/hadamard.rs` | 1 | 12 | `apeiron-lane-1.lock` | 540/525/530 |
| [`Gguf`](../../proof/resource/Gguf_gate.py) | yes | `e9f88888e0d4` | `GGUF_SOURCE_RESOURCE_OK` | `src/format/gguf.rs` | 2 | 29 | `apeiron-lane-2.lock` | 540/525/530 |
| [`I64Codec`](../../proof/resource/I64Codec_gate.py) | yes | `158a6afecb99` | `I64_CODEC_SOURCE_RESOURCE_OK` | `src/table/keyrep.rs` | 1 | 14 | `apeiron-lane-3.lock` | 540/525/530 |
| [`I64Decode`](../../proof/resource/I64Decode_gate.py) | yes | `5f4e43a29185` | `I64_DECODE_SOURCE_RESOURCE_OK` | `src/table/keyrep.rs` | 1 | 27 | `apeiron-lane-2.lock` | 540/525/530 |
| [`KvAppend`](../../proof/resource/KvAppend_gate.py) | no | `5a6039d0c0f3` | `KV_APPEND_SOURCE_RESOURCE_OK` | `src/engine/attn.rs` | 1 | 0 | `apeiron-lane-1.lock` | 540/525/530 |
| [`LmHead`](../../proof/resource/LmHead_gate.py) | no | `0b9cc8f5d9c6` | `LM_HEAD_SOURCE_RESOURCE_OK` | `src/engine/matvec.rs` | 1 | 0 | `apeiron-lane-1.lock` | 540/525/530 |
| [`Matvec`](../../proof/resource/Matvec_gate.py) | no | `4b761ea3c69f` | `MATVEC_SOURCE_RESOURCE_OK` | `src/engine/matvec.rs` | 1 | 20 | `apeiron-lane.lock` | 540/525/530 |
| [`Packing`](../../proof/resource/Packing_gate.py) | yes | `5f90a33913c2` | `PACKING_SOURCE_RESOURCE_OK` | `src/format/packing.rs` | 1 | 16 | `apeiron-lane.lock` | 540/525/530 |
| [`Policy`](../../proof/resource/Policy_gate.py) | yes | `b8eb08ec02a8` | `POLICY_SOURCE_RESOURCE_OK` | `src/policy/decide.rs` | 1 | 18 | `apeiron-lane-3.lock` | 540/525/530 |
| [`PublishPrepared`](../../proof/resource/PublishPrepared_gate.py) | yes | `f9fb9b6c61df` | `PUBLISH_PREPARED_SOURCE_RESOURCE_OK` | `src/table/resources.rs` | 1 | 22 | `apeiron-lane-1.lock` | 540/525/530 |
| [`Q4KCodes`](../../proof/resource/Q4KCodes_gate.py) | yes | `6937c678fb67` | `Q4K_CODES_SOURCE_RESOURCE_OK` | `src/format/quant.rs` | 1 | 20 | `apeiron-lane-1.lock` | 540/525/530 |
| [`Q6KCodes`](../../proof/resource/Q6KCodes_gate.py) | yes | `6c0f8ff06434` | `Q6K_CODES_SOURCE_RESOURCE_OK` | `src/format/quant.rs` | 2 | 18 | `apeiron-lane-2.lock` | 540/525/530 |
| [`RegenRelocateAll`](../../proof/resource/RegenRelocateAll_gate.py) | no | `4aefb25b3a59` | `REGEN_RELOCATE_ALL_SOURCE_RESOURCE_OK` | `src/table.rs` | 1 | 0 | `apeiron-lane-2.lock` | 540/525/530 |
| [`RegenRelocateEntry`](../../proof/resource/RegenRelocateEntry_gate.py) | no | `a44144551c22` | `REGEN_RELOCATE_ENTRY_SOURCE_RESOURCE_OK` | `src/table.rs` | 1 | 20 | `apeiron-lane-3.lock` | 540/525/530 |
| [`RelocateEntry`](../../proof/resource/RelocateEntry_gate.py) | yes | `d748b7c57cf1` | `RELOCATE_ENTRY_SOURCE_RESOURCE_OK` | `src/table.rs` | 1 | 12 | `apeiron-lane-3.lock` | 540/525/530 |
| [`RmsNorm`](../../proof/resource/RmsNorm_gate.py) | yes | `4e2d1b45cd36` | `RMS_NORM_SOURCE_RESOURCE_OK` | `src/engine/norm.rs` | 1 | 21 | `apeiron-lane.lock` | 540/525/530 |
| [`Rope`](../../proof/resource/Rope_gate.py) | yes | `568c56f54d2d` | `ROPE_SOURCE_RESOURCE_OK` | `src/engine/rope.rs` | 1 | 7 | `apeiron-lane-2.lock` | 540/525/530 |
| [`RouteTopk`](../../proof/resource/RouteTopk_gate.py) | yes | `24fac87eb7b8` | `ROUTE_TOPK_SOURCE_RESOURCE_OK` | `src/engine/moe.rs` | 1 | 22 | `apeiron-lane-2.lock` | 540/525/530 |
| [`RowDotAdmission`](../../proof/resource/RowDotAdmission_gate.py) | yes | `5e3ed2929b97` | `ROW_DOT_ADMISSION_COST_CERTIFICATE_OK` | `proof/resource/RowDotAdmission.lean` | 2 | 48 | `apeiron-lane.lock` | 540/525/530 |
| [`RowDot`](../../proof/resource/RowDot_gate.py) | yes | `2a8f077abf2f` | `ROW_DOT_SOURCE_RESOURCE_OK` | `src/engine/matvec.rs` | 1 | 20 | `apeiron-lane.lock` | 540/525/530 |
| [`Schedule`](../../proof/resource/Schedule_gate.py) | yes | `458f33af1682` | `SCHEDULE_SOURCE_RESOURCE_OK` | `src/table/partition.rs` | 1 | 31 | `apeiron-lane-2.lock` | 540/525/530 |
| [`Silu`](../../proof/resource/Silu_gate.py) | yes | `019c6b0fb0ec` | `SILU_SOURCE_RESOURCE_OK` | `src/engine/silu.rs` | 1 | 20 | `apeiron-lane-1.lock` | 540/525/530 |
| [`StoredRelocateAll`](../../proof/resource/StoredRelocateAll_gate.py) | yes | `845520ce9325` | `STORED_RELOCATE_ALL_SOURCE_RESOURCE_OK` | `src/table.rs` | 1 | 33 | `apeiron-lane-1.lock` | 540/525/530 |

## Deviations from the uniform shape

Recorded, not repaired. Each names the command that shows it.

- **`RmsNorm`** asserts its needle occurs **2** times, not once: `i += 1;` in `src/engine/norm.rs` (`proof/resource/RmsNorm_gate.py`). Every other row asserts `count(needle) != 1`.
- **`KvAppend`** asserts `KV_APPEND_KERNEL_OK`, but the resource file its script copies and builds, `proof/resource/KvAppendResource.lean`, is not on disk. The gate cannot pass as written.
- **`LmHead`** asserts `LM_HEAD_KERNEL_OK`, but `proof/resource/LmHeadResource.lean` carries no `#resource_audit` line and no such marker, so the build output can never contain it. The gate cannot pass against the committed file.
- **`RegenRelocateAll`** asserts `REGEN_RELOCATE_ALL_KERNEL_OK`, but the resource file its script copies and builds, `proof/resource/RegenRelocateAllResource.lean`, is not on disk. The gate cannot pass as written.
- Multi-control rows: `Gguf` (2), `Q6KCodes` (2), `RowDotAdmission` (2).
- Rows whose script substitutes a production body before extracting, recorded separately from the seeded defect: `RowDotAdmission`.

## Gate

Three properties, one bounded run, no committed script. Exits 0 with
`CONFORMANCE_OK rows=<n>`: each row's `sha256` equals its script on disk, each
`mutation[].needle` occurs `mutation[].occurrences` times in the file that row mutates,
and every `proof/resource/*_gate.py` has a row.

```sh
python3 - <<'PY'
import hashlib, json, pathlib, sys
ROOT = pathlib.Path('/home/socol/Workspace/apeiron')
REC = ROOT / 'tools/gate/conformance.json'
rec = json.loads(REC.read_text())
bad = []
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(1 << 16), b''):
            h.update(c)
    return h.hexdigest()
for g in rec['gates']:
    if sha(ROOT / g['path']) != g['sha256']:
        bad.append(f"{g['id']}: sha256 drift on {g['path']}")
    for m in g['mutation']:
        n = (ROOT / m['file']).read_text().count(m['needle'])
        if n != m['occurrences']:
            bad.append(f"{g['id']}/{m['control']}: needle in {m['file']} occurs {n}, "
                       f"record says {m['occurrences']}")
on_disk = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'proof/resource').glob('*_gate.py')}
recorded = {g['path'] for g in rec['gates']}
for p in sorted(on_disk - recorded):
    bad.append(f'MISSING_ROW {p}')
for p in sorted(recorded - on_disk):
    bad.append(f'STALE_ROW {p}')
if bad:
    print('\n'.join(bad))
    sys.exit(1)
print(f"CONFORMANCE_OK rows={len(rec['gates'])}")
PY
```

## Not established

- The record freezes what each script already carries. It runs no gate, so it establishes nothing about the correctness of any theorem.
- The sha256 values bind this record to the scripts on disk at generation time. generated_at_head is a snapshot that drifts as other tasks commit; it is not a gate input.
- No script here was executed. green_marker and red_marker are transcribed from the scripts, not observed from a run.

