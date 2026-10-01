#!/usr/bin/env python3
"""Source-bound candidate-layout/state model checks; no Rust/GPU refinement claim.

O(B) source bytes plus O(M*K + padded_batch*K) witness lane/address work.
"""
import argparse, copy, hashlib, json, math, struct, time
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
FOLD=Path('/home/socol/Workspace/fold')
EXPECTED={
'src/format/quant.rs':'e574ea15520ed3c17dbfb90d514c2134a3883f681bbfba63b1eb200992f65335',
'src/engine/chart/mod.rs':'ae4a4a247f01b8435cb93860554d4b7d795b4c41d804b1d27abe614f23ad68c0',
'src/engine/kern/gemm.rs':'44127bd20c1b54ce058b97df16529bc6346a1e22661a45916abbf16dad29d444',
'modelfold/src/simt/dp4a.rs':'ffe4f6c313492b84ecfc2c5d99c9fa8b2945f5bf081282e9b13b1374da50901e',
'modelfold/src/simt/superblock.rs':'e15e18bacfa02b5508dfefc959fc9bd7f2ef1593ff208d8de6e5e3c716745387',
'modelfold/src/simt_nn/spec.rs':'f54e9fe3590aa4c6c87d382a8a4c9b76a56376eda58d212e4f2dd725a2c445b0',
'bedrock/src/serve/model/pack.rs':'b927d380a78eb283abe7ef47631a3d33de48a700febc6fd67f7cc0a8a65e3211',
'bedrock/src/serve/gpu.rs':'1e01838dca24f7751274249af671bb485109b80f8f0e7f9fa940e73744e9b17c',
'modelfold/src/simt_nn/batch.rs':'d183f9cb8decb9ffccb37c91a7973c39003b33361babc8dc6f98615cfc618fd7',
'modelfold/src/simt_nn/deltanet.rs':'c72f1859e82a974339b208ccf6ae67455fa7be73b85f6845887c5d6b69906879'}

def require(ok,reason):
    if not ok: raise ValueError(reason)

def exact_float(q,fmt):
    """Exact finite representation predicate, including overflow and nonzero underflow."""
    try: f=struct.unpack(fmt,struct.pack(fmt,float(q)))[0]
    except (OverflowError,struct.error): return False
    return math.isfinite(f) and Fraction(f)==q

def recode(d,codes):
    if not exact_float(d,'e') or not exact_float(-128*d,'e'): return None
    return [q+128 for q in codes],d,-128*d

def address(order,a,b,first,second):
    return a*second+b if order=='first-major' else b*first+a

def continuation(policy,owner,request):
    return all(owner[k]==request[k] for k in policy)

def semantics(data):
    """Evaluate candidate numeric/layout/state choices against independent observations."""
    m=data['model'];rows,k,b,bt=3,64,5,4
    require(m['weight_offset']==128,'SIGNED_WEIGHT_RECODE')
    q=[-128,-7,0,9,127];x=[-127,21,0,-3,64];d=Fraction(3,8)
    candidate=recode(d,q);require(candidate is not None,'F16_FINITE_WITNESS')
    u,ds,lo=candidate
    require(ds*sum(a*y for a,y in zip(u,x))+lo*sum(x)==d*sum(a*y for a,y in zip(q,x)),'AFFINE_RECODE')
    require(recode(Fraction(512),q) is None,'F16_OFFSET_OVERFLOW_REFUSAL')
    require(recode(Fraction(1,2**30),q) is None,'F16_INEXACT_REFUSAL')
    require(exact_float(Fraction(1,2**32),'f') and not exact_float(Fraction(2**32+1,2**32),'f'),'F32_EXACT_REFUSAL')
    require(m['weight_order'] in ('first-major','second-major') and m['dm_order'] in ('first-major','second-major'),'LAYOUT_ORDER')
    for word in range(k//4):
        for row in range(rows):
            require(address(m['weight_order'],word,row,k//4,rows)==word*rows+row,'WEIGHT_STRIDE')
    for block in range(k//32):
        for row in range(rows):
            require(address(m['dm_order'],block,row,k//32,rows)==block*rows+row,'DM_STRIDE')
    bp=((b+bt-1)//bt)*bt if m['pad_tokens'] else b
    require(bp==8 and bp%bt==0,'PADDED_EXTENT')
    values=[[((i+j)%255)-127 for j in range(k)] for i in range(b)]
    values += [[0]*k for _ in range(bp-b)]
    packed=[[struct.pack('<bbbb',*v[j:j+4]) for j in range(0,k,4)] for v in values]
    for token in range(bp):
        for j in range(k):
            got=struct.unpack('<bbbb',packed[token][j//4])[j%4]
            require(got==values[token][j] and (token<b or got==0),'SIGNED_ACTIVATION_LANE')
    observed=bp if m['observe_padded'] else b
    require(observed==b,'PADDED_OUTPUT_OBSERVATION')
    scale=Fraction(1,8)
    sx=scale if m['sx_source']=='quantact-scale' else Fraction(1)
    reconstructed=[v*scale for v in x]
    original=[v+Fraction(1,256) for v in reconstructed]
    xs=sum(reconstructed if m['xs_source']=='reconstructed' else original)
    require(sx==scale and xs==sum(reconstructed),'XS_SX_RECONSTRUCTION')
    owner={'model':'a'*64,'generation':17,'prefix':64,'capacity':512}
    fields=m['continuation_fields']
    require(set(fields)==set(owner) and len(fields)==len(owner),'SESSION_GUARD_DOMAIN')
    require(continuation(fields,owner,owner.copy()),'SESSION_POSITIVE')
    for key in owner:
        foreign=owner.copy();foreign[key]='b'*64 if key=='model' else owner[key]+1
        require(not continuation(fields,owner,foreign),'FOREIGN_SESSION_PREFIX')
    reset={'generation':18,'kv':[0]*4,'recurrent':[0]*4}
    require(reset['generation']!=owner['generation'] and not any(reset['kv']+reset['recurrent']),'RESET_WITNESS')
    require(k%32==0 and 33%32!=0 and b>0,'SHAPE_REFUSAL')
    require((2**64-1)//3<2**64-1 and (2**64-1)*3>2**64-1,'SIZE_OVERFLOW_REFUSAL')
    require(m['empty_policy']=='refuse' and m['overflow_policy']=='refuse','REFUSAL_POLICY')

def validate(data,live=True):
    sources=data.get('sources',[])
    require(len(sources)==len(EXPECTED) and {s['path'] for s in sources}==set(EXPECTED),'SOURCE_SET_INCOMPLETE')
    for item in sources:
        rel=item['path'];require(item['sha256']==EXPECTED[rel],'SOURCE_DIGEST')
        if live:
            p=(ROOT if rel.startswith('src/') else FOLD)/rel
            require(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'LIVE_SOURCE_DIGEST')
            require(bool(item['symbols']) and all(x in p.read_text() for x in item['symbols']),'SOURCE_SYMBOL')
    require(data['verification']['gate_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'GATE_DIGEST')
    c=data['contract']
    require(c['bindings']==['W','XQ','OUT','DM','XS','SX'],'BINDINGS')
    require(c['dm_address']=='DM[block*M+row], packed f16 (d,lo)','DM_CONTRACT')
    require(c['xs_sx_address']=='XS[token*blocks+block], SX[token*blocks+block]','XS_SX_ADDRESS')
    require(c['activation_gpu_encoding']=='Direct signed QuantAct XQ; exact QuantAct scale as SX; XS from reconstructed lanes only; no requantization','XS_SX_CONTRACT')
    require(c['continuation']=='same model digest, session generation, prefix and capacity; otherwise reset KV and recurrent state','CONTINUATION_CONTRACT')
    semantics(data)

def controls(data):
    changes=[('empty-sources',lambda d:d.update(sources=[])),
      ('wrong-source',lambda d:d['sources'][0].update(sha256='0'*64)),
      ('signedness',lambda d:d['model'].update(weight_offset=0)),
      ('weight-stride',lambda d:d['model'].update(weight_order='second-major')),
      ('dm-stride',lambda d:d['model'].update(dm_order='second-major')),
      ('padded-extent',lambda d:d['model'].update(pad_tokens=False)),
      ('padded-observation',lambda d:d['model'].update(observe_padded=True)),
      ('original-input-xs',lambda d:d['model'].update(xs_source='original')),
      ('wrong-sx',lambda d:d['model'].update(sx_source='constant-one')),
      ('foreign-generation',lambda d:d['model'].update(continuation_fields=['model','prefix','capacity'])),
      ('stale-prefix',lambda d:d['model'].update(continuation_fields=['model','generation','capacity'])),
      ('old-dm-contract',lambda d:d['contract'].update(dm_address='DM[row*blocks+block]')),
      ('old-xs-contract',lambda d:d['contract'].update(activation_gpu_encoding='signed i8 XQ; XS=sum(original input), SX=1')),
      ('old-state-contract',lambda d:d['contract'].update(continuation='reuse foreign generation and stale prefix'))]
    for label,mutate in changes:
        bad=copy.deepcopy(data);mutate(bad)
        try: validate(bad,True)
        except ValueError as e: print('CONTROL_RED',label,str(e))
        else: raise AssertionError('CONTROL_FALSE_GREEN:'+label)
        validate(copy.deepcopy(data),True);print('CONTROL_RESTORED',label)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--controls',action='store_true');args=parser.parse_args()
    start=time.monotonic();data=json.loads((HERE/'adapter-identities.json').read_text());validate(data)
    print('SOURCE_BOUND_MODEL_CHECK_PASS')
    if args.controls: controls(data)
    print('GRADE source-bound model; checker tested; Rust/GPU refinement OPEN',f'elapsed={time.monotonic()-start:.3f}')
