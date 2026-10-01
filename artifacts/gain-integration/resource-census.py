#!/usr/bin/env python3
"""Source-bound resource MODEL; integrity controls do not prove Rust/SPIR-V cost correspondence."""
import argparse
import copy
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAIN = Path('/home/socol/Workspace/apeiron')
FOLD = Path('/home/socol/Workspace/fold')
RETAINED = '04d206013f70f245dfdaf2619c0b1846d10b9922'
CPU = ['src/engine/step/cols.rs', 'src/engine/step.rs', 'src/engine/kern/matmat.rs',
       'src/engine/kern/gemm.rs', 'src/engine/chart/pool.rs', 'src/engine/chart/ffn.rs',
       'src/engine/chart/mod.rs', 'src/engine/loader.rs', 'src/format/quant.rs',
       'src/engine/kern/pow2.rs']
GPU = ['modelfold/src/simt/superblock.rs', 'modelfold/src/simt/dp4a.rs',
       'modelfold/src/simt_nn/spec.rs', 'modelfold/src/simt_nn/batch.rs',
       'modelfold/src/simt_nn/deltanet.rs', 'modelfold/src/gpu/runtime.rs',
       'bedrock/src/serve/gpu.rs']
RECEIPTS = ['1790795647-w6e-vs-llama-r3.json', '1790798160-w7-q35r2-pp512.json',
            '1790798706-w7-qwen05b-r2-pp200.json', '1790798386-w7-qwen15b-r5-pp200.json']


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def model_directory(path):
    """Read actual GGUF metadata/tensor directory; never read weight payloads."""
    with Path(path).open('rb') as f:
        def scalar(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize(fmt)))[0]
        def string():
            return f.read(scalar('Q')).decode('utf-8')
        def value(t):
            if t == 8:
                return string()
            if t == 9:
                typ, n = scalar('I'), scalar('Q')
                return [value(typ) for _ in range(n)]
            return scalar({0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                           6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}[t])
        assert f.read(4) == b'GGUF', 'MODEL_HEADER'
        version, nt, nm = scalar('I'), scalar('Q'), scalar('Q')
        assert version in (2, 3), 'MODEL_VERSION'
        meta = {}
        for _ in range(nm):
            name = string()
            meta[name] = value(scalar('I'))
        tensors = []
        for _ in range(nt):
            name = string()
            dims = [scalar('Q') for _ in range(scalar('I'))]
            typ, offset = scalar('I'), scalar('Q')
            tensors.append({'name': name, 'dims': dims, 'ggml_type': typ, 'offset': offset})
        end = f.tell()
        f.seek(0)
        digest = sha(f.read(end))
    arch = meta['general.architecture']
    keys = ['block_count', 'nextn_predict_layers', 'embedding_length', 'feed_forward_length',
            'attention.head_count', 'attention.head_count_kv', 'attention.key_length',
            'full_attention_interval', 'ssm.state_size', 'ssm.group_count', 'ssm.time_step_rank']
    selected = {k: meta.get(arch + '.' + k) for k in keys}
    return {'path': str(path), 'directory_sha256': digest, 'directory_bytes': end,
            'architecture': arch, 'metadata': selected, 'tensors': tensors}


def block_sizes(n, width=256):
    count = (n + width - 1) // width
    base, rem = divmod(n, count)
    return [base + (i < rem) for i in range(count)]


def cpu_group(ms, k, b):
    """GEMM pack + row dispatch on its successful branch, plus quant_acts_pooled copy."""
    assert b > 1 and k % 32 == 0, 'SHAPE_DOMAIN'
    blocks, groups = k // 32, (b + 7) // 8
    words = groups * blocks * 8 * 8 * 4
    shifts = groups * blocks * 8 * 8
    emin, spread = b * 4, groups * 4
    rpt = 4 if len(ms) == 1 and ms[0] <= 1024 else 16
    quant_bytes = b * (k + blocks * 8)
    return {'grade': 'model', 'rows': ms, 'n_in': k, 'columns': b, 'activation_blocks': blocks,
            'column_groups': groups, 'weight_payload_distinct_bytes': sum(ms) * blocks * 34,
            'scalar_mac_terms': sum(ms) * k * b, 'output_bytes': sum(ms) * b * 8,
            'packing_words_bytes': words, 'packing_shift_bytes': shifts,
            'packing_emin_bytes': emin, 'packing_spread_bytes': spread,
            'packing_write_bytes': words + shifts + emin + spread,
            'packing_code_read_bytes': b * k, 'packing_scale_read_bytes': b * blocks * 16,
            'packing_named_buffer_allocations': 4, 'packing_pool_dispatches': 1,
            'row_pool_dispatches': 1, 'row_pool_tasks': sum((m + rpt - 1) // rpt for m in ms),
            'quantization_pool_dispatches': 1, 'quantization_copy_bytes': quant_bytes,
            'quantization_named_buffer_allocations': 4 * b,
            'quantization_persistent_bytes': quant_bytes,
            'quantization_input_lane_visits': 2 * b * k,
            'row_code_passes_per_full_row_pair': (groups + 1) // 2,
            'weight_load_instructions': 'UNKNOWN: path-dependent shift/exact fallback and odd rows',
            'hardware_dram_bytes': 'UNKNOWN', 'whole_operator_allocations': 'UNKNOWN',
            'whole_operator_peak_live_bytes': 'UNKNOWN',
            'named_pack_plus_quant_output_bytes': words + shifts + emin + spread + quant_bytes,
            'critical_path_time': 'UNKNOWN'}


def tile(m, k, pack=4, local=128):
    chart = 4 * m * (k // pack + k // 32)
    bt, rt = (32 if chart > 32 << 20 else 8), 1
    rt = 64 // bt
    while rt > 1:
        slabs = (m + local * rt - 1) // (local * rt)
        if slabs >= 2 and (slabs * local * rt - m) * 2 <= m:
            break
        rt //= 2
    return bt, rt


def gpu_kernel(m, k, b, pack=4, local=128, bt=None, rt=None):
    assert m > 0 and k > 0 and k % 32 == 0 and pack in (4, 8, 16), 'SHAPE_DOMAIN'
    if bt is None:
        bt, rt = tile(m, k, pack, local)
    assert bt > 0 and rt > 0, 'TILE_DOMAIN'
    bp = ((b + bt - 1) // bt) * bt
    gx = (m + local * rt - 1) // (local * rt)
    inv, blocks, cpb = gx * local * (bp // bt), k // 32, 32 // pack
    loads = inv * blocks * (rt * (1 + cpb) + 10 * bt)
    return {'grade': 'model', 'carrier': {4: 'Q8', 8: 'Q4', 16: 'Q2'}[pack],
            'm': m, 'k': k, 'batch': b, 'padded_batch': bp, 'bt': bt, 'rt': rt, 'local': local,
            'workgroups': [gx, bp // bt], 'invocations': inv,
            'global_load_words': loads, 'logical_global_load_bytes': 4 * loads,
            'integer_dot4_instructions': inv * blocks * 8 * rt * bt,
            'float_multiplies': inv * blocks * 3 * rt * bt,
            'float_adds': inv * blocks * 2 * rt * bt,
            'valid_output_stores': m * bp, 'chart_distinct_bytes': 4 * m * (k // pack + blocks),
            'projection_upload_bytes': bp * k + 8 * bp * blocks,
            'projection_readback_bytes': b * m * 4, 'projection_queue_drains': 1,
            'hardware_dram_bytes': 'UNKNOWN', 'gpu_executed_path': 'UNKNOWN: no GPU run in this task'}


def source_rows():
    rows = []
    for rel in CPU:
        retained = git('show', RETAINED + ':' + rel)
        current = (ROOT / rel).read_bytes()
        rows.append({'tree': 'apeiron', 'path': rel, 'retained_revision': RETAINED,
                     'retained_sha256': sha(retained), 'current_sha256': sha(current),
                     'current_equals_retained': current == retained})
    fold_rev = subprocess.check_output(['git', '-C', str(FOLD), 'rev-parse', 'HEAD']).decode().strip()
    for rel in GPU:
        rows.append({'tree': 'fold', 'path': rel, 'revision': fold_rev,
                     'current_sha256': sha((FOLD / rel).read_bytes())})
    return rows


def groups_for(model):
    md = model['metadata']
    layers = md['block_count'] - (md['nextn_predict_layers'] or 0)
    ts = {t['name']: t for t in model['tensors']}
    def get(name):
        t = ts[name]
        assert t['ggml_type'] == 8 and len(t['dims']) == 2, 'MODEL_Q8_SHAPE'
        k, m = t['dims']
        return m, k
    out = []
    for layer in range(layers):
        prefix = f'blk.{layer}.'
        affine = prefix + 'attn_qkv.weight' in ts
        inputs = (['attn_qkv', 'attn_gate', 'ssm_alpha', 'ssm_beta'] if affine
                  else ['attn_q', 'attn_k', 'attn_v'])
        names = [[x + '.weight' for x in inputs],
                 [('ssm_out' if affine else 'attn_output') + '.weight'],
                 ['ffn_gate.weight', 'ffn_up.weight'], ['ffn_down.weight']]
        for kind, ns in zip(['affine_input' if affine else 'qkv', 'mixer_output', 'ffn_gate_up', 'ffn_down'], names):
            shapes = [get(prefix + n) for n in ns]
            assert len({k for _, k in shapes}) == 1, 'GROUP_WIDTH'
            out.append({'kind': kind, 'layer': layer, 'rows': [m for m, _ in shapes],
                        'k': shapes[0][1], 'tensor_names': [prefix + n for n in ns]})
    return layers, out


def build():
    evidence = MAIN / 'artifacts/track-consolidation'
    manifest = json.loads((evidence / 'source-manifest.json').read_text())
    ratios = json.loads((evidence / 'archived-ratios.json').read_text())
    sources = source_rows()
    for row in sources:
        if row['tree'] == 'apeiron' and row['path'] in manifest['combined_source_sha256']:
            row['combined_manifest_sha256'] = manifest['combined_source_sha256'][row['path']]
            row['matches_combined_manifest'] = row['current_sha256'] == row['combined_manifest_sha256']
    workloads = []
    for name in RECEIPTS:
        path = evidence / 'evidence/ab' / name
        raw = json.loads(path.read_text())
        rr = next(r for r in ratios['rows'] if r['receipt'].endswith(name))
        assert raw['protocol']['pp'] == rr['pp'], 'RECEIPT_SHAPE'
        md = model_directory(raw['identity']['model'])
        layers, groups = groups_for(md)
        shapes = {}
        for g in groups:
            key = (g['kind'], tuple(g['rows']), g['k'])
            shapes.setdefault(key, {'multiplicity': 0, 'group': g})['multiplicity'] += 1
        bs = block_sizes(rr['pp'])
        census = [{'kind': key[0], 'multiplicity': v['multiplicity'],
                   'tensor_example': v['group']['tensor_names'],
                   'blocks': [cpu_group(list(key[1]), key[2], b) for b in bs],
                   'gpu_projection_examples': [gpu_kernel(sum(key[1]), key[2], b) for b in bs]}
                  for key, v in sorted(shapes.items())]
        for entry in census:
            if entry['kind'] == 'ffn_gate_up':
                for cost in entry['blocks']:
                    ff = cost['rows'][0]
                    cost['row_pool_tasks'] = (ff + 15) // 16
                    cost['output_bytes'] = ff * cost['columns'] * 8
                    cost['avoided_gate_up_intermediate_bytes'] = 2 * ff * cost['columns'] * 8
                    cost['path_condition'] = 'fused gemm_q8_0_swiglu_carrier successful; fallback UNKNOWN'
        hybrid = any(g['kind'] == 'affine_input' for g in groups)
        h, nh, nkv = md['metadata']['embedding_length'], md['metadata']['attention.head_count'], md['metadata']['attention.head_count_kv']
        hd = md['metadata']['attention.key_length'] or h // nh
        embed = next(t for t in md['tensors'] if t['name'] == 'token_embd.weight')
        vocab = embed['dims'][1]
        resident = {'coverage': 'hybrid batched recurrence absent' if hybrid else 'existing dense Fold graph; adapter/execution unrun',
                    'cpu_state_owner_transition': 'same owner + dev_len>=pos0: no seed; changed owner: prefix seed; gap: missing suffix seed',
                    'hybrid_decode': 'existing recurrent buffers+reset; batched scheduling UNKNOWN',
                    'constructor_allocation_bytes': 'UNKNOWN', 'peak_live_bytes': 'UNKNOWN',
                    'cold_weight_upload_bytes': 'UNKNOWN: Fold chart construction/loading not executed',
                    'actual_gpu_path': 'UNKNOWN', 'blocks': []}
        if not hybrid:
            for b in bs:
                resident['blocks'].append({'b': b, 'all_dispatches': 11 * layers + 2,
                    'last_dispatches': 11 * layers + 3, 'continuation_queue_drains': 1,
                    'embedding_and_params_upload_bytes': 4 * b * h + 16,
                    'all_hidden_logits_host_visible_write_bytes': 4 * b * (h + vocab),
                    'last_hidden_logits_host_visible_write_bytes': 4 * b * h + 4 * tile(vocab, h)[0] * vocab,
                    'kv_host_visible_write_bytes': 8 * layers * b * nkv * (hd + 1),
                    'kv_mirror_host_write_bytes': 2 * layers * b * nkv * hd + 8 * layers * b * nkv,
                    'seed_upload_bytes_for_gap_per_position': 8 * layers * nkv * (hd + 1),
                    'seed_extra_drains': 'UNKNOWN: staging chunks/owner/prefix dependent'})
        traces = [t for t in raw.get('traces', []) if t['workload'] == 'pp']
        workloads.append({'receipt': str(path), 'receipt_sha256': sha(path.read_bytes()),
            'model_directory': {k: v for k, v in md.items() if k != 'tensors'},
            'model_payload_sha256_historical': raw['identity'].get('model_sha256'),
            'model_payload_freshly_authenticated': False, 'pp': rr['pp'], 'tg': rr['tg'],
            'trunk_layers': layers, 'block_sizes': bs, 'hybrid': hybrid,
            'retained_ratio': rr, 'receipt_identity': raw['identity'],
            'historical_pp_traces': traces, 'cpu_success_branch_model': census,
            'gpu_resident_model': resident,
            'cpu_current_path_witness': 'UNKNOWN: no inference executed; archived trace is historical phase witness'})
    return {'schema': 'apeiron.resource-census.v1', 'grade': 'model', 'source_identity': sources,
        'evidence': [{'path': str(evidence / p), 'sha256': sha((evidence / p).read_bytes())}
                     for p in ['source-manifest.json', 'archived-ratios.json']],
        'workloads': workloads, 'held_out': held_out(),
        'proof_target': {'path': 'src/format/quant.rs', 'body': 'q8_0_codes',
            'existing_theorem': 'apeiron_word.q8_0_codes_ok in Numeric/DequantQ8_0.lean',
            'successful_loop': '32 iterations, 32 byte reads, 32 signed casts, 32 array writes',
            'refusals': 'checked_mul overflow or truncated block: zero loop iterations; o+34 addition requires no-overflow premise',
            'scope': 'scalar accessor; does not prove SIMD/pool/emitter/whole-model resource cost'}}


def held_out():
    # Frozen shape not present in the four workload projection directories.
    baseline = gpu_kernel(3072, 2048, 384, bt=8, rt=1)
    tiled = gpu_kernel(3072, 2048, 384, bt=8, rt=4)
    reduction = 1 - tiled['global_load_words'] / baseline['global_load_words']
    return {'grade': 'model', 'shape': {'m': 3072, 'k': 2048, 'b': 384, 'carrier': 'Q8', 'local': 128},
        'baseline': baseline, 'candidate': tiled, 'logical_load_reduction_range': [reduction, reduction],
        'dot_and_store_counts_preserved': baseline['integer_dot4_instructions'] == tiled['integer_dot4_instructions'] and baseline['valid_output_stores'] == tiled['valid_output_stores'],
        'packing_bytes_change': 0, 'dispatch_count_change': 0,
        'falsifier': 'Actual emitted per-binding loads differ, dot/store extents differ, or adapter adds a repack/transfer; hardware speed is not predicted'}


def validate(record, expected):
    if record['source_identity'] != expected['source_identity']:
        raise ValueError('SOURCE_BINDING')
    if record['evidence'] != expected['evidence']:
        raise ValueError('EVIDENCE_BINDING')
    for got, want in zip(record['workloads'], expected['workloads']):
        if got['model_directory'] != want['model_directory'] or got['pp'] != want['pp'] or got['block_sizes'] != want['block_sizes']:
            raise ValueError('SHAPE_BINDING')
        if got['cpu_success_branch_model'] != want['cpu_success_branch_model']:
            raise ValueError('PACKING_OR_COST')
    if record != expected:
        raise ValueError('RECORD_DRIFT')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    expected = build()
    if args.write:
        (HERE / 'resource-census.json').write_text(json.dumps(expected, indent=2) + '\n')
    record = json.loads((HERE / 'resource-census.json').read_text())
    validate(record, expected)
    print('PASS baseline: 17 live/retained sources, 4 GGUF workload directories, packing/tile/graph counts match')
    if args.controls:
        def reject(label, change, diagnostic):
            mutant = copy.deepcopy(record)
            change(mutant)
            try:
                validate(mutant, expected)
            except ValueError as e:
                assert str(e) == diagnostic, (label, str(e))
                print(f'PASS control {label}: rejected {e}')
            else:
                raise AssertionError('CONTROL_SURVIVED: ' + label)
            validate(record, expected)
            print(f'PASS restore {label}')
        reject('wrong-source', lambda r: r['source_identity'][0].update(current_sha256='0' * 64), 'SOURCE_BINDING')
        reject('wrong-shape', lambda r: r['workloads'][0].update(pp=201), 'SHAPE_BINDING')
        reject('omitted-packing', lambda r: r['workloads'][0]['cpu_success_branch_model'][0]['blocks'][0].update(packing_write_bytes=0), 'PACKING_OR_COST')
        pred = record['held_out']
        assert pred['dot_and_store_counts_preserved']
        print('PASS held-out: exact logical load reduction %.9f; equal dots/stores' % pred['logical_load_reduction_range'][0])
    print('GRADE model; checker/control execution tested; no Rust/SPIR-V proof or hardware measurement')
    print('elapsed_seconds=%.6f' % (time.monotonic() - start))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, AssertionError, KeyError) as error:
        print('FAIL: ' + str(error), file=sys.stderr)
        sys.exit(1)
