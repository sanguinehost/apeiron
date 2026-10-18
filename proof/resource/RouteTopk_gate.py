#!/usr/bin/env python3
"""Extract and check the real MoE top-k router source in a bounded private build.

Adapted from proof/resource/gate.py (and its row-dot sibling) for the extraction root
`crate::word_route_topk`, the same root proof/regen.sh names. O(B) source/tool/log bytes;
external compilation is timed, not assigned constant cost. The temporary manifest wires the
real `src/engine/{matvec,value,error,dequant,f32_convert,ffn,par,moe}.rs`,
`src/format/{quant,gguf}.rs` and `src/numeric/fixed_exp.rs` by `#[path]`; the only new code is
the typed forwarding wrapper that calls the real `route_topk` body. No substitute algorithm is
extracted, and the resource measure is a scalar comparison/scan count, never a cycle or GPU time.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path(os.environ.get('APEIRON_TOOLS') or Path(__file__).resolve().parents[2])
LANE = '/tmp/opencode/apeiron-lane-2.lock'
if __name__ == '__main__' and os.environ.get('ROUTE_TOPK_IN_LANE') != '1':
    deadline = str(time.time()+535)
    os.execvp('timeout', ['timeout','--kill-after=5','540s','flock','-w','530',LANE,
        'systemd-run','--user','--scope','-q','-p','MemoryMax=12G','-p','MemorySwapMax=0',
        'timeout','--kill-after=5','525s','env','LEAN_NUM_THREADS=1','ROUTE_TOPK_IN_LANE=1',
        'ROUTE_TOPK_DEADLINE='+deadline,sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
START = time.monotonic()
RECORDS = []
# Scratch copies live off the tmpfs (AGENTS.md failure register: /tmp tmpfs headroom); only named
# receipts and logs are kept under /tmp/opencode.
OUT = Path('/tmp/opencode/routetopk-resource')
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_ROOT = Path(tempfile.gettempdir()) / 'routetopk-resource'
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(tempfile.mkdtemp(prefix='source-', dir=SCRATCH_ROOT))
# The finite visit-count certificate, read from the repository's own proved file
# (proof/native/LoopTrace.lean): `WP.loop.spec_decr_nat` carries no count, so the count is
# supplied by that certificate and projected to the actual `loop`.
LOOPTRACE = ROOT/'proof/native/LoopTrace.lean'

def digest(p):
    """O(B) file bytes, O(1) streaming scratch for file size B."""
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(argv, cwd, label, require=True):
    """Run within the shared whole-invocation deadline; record complete output."""
    left = min(525 - (time.monotonic()-START), float(os.environ.get('ROUTE_TOPK_DEADLINE', time.time()+525))-time.time()-3)
    if left <= 0:
        raise RuntimeError('ROUTE_TOPK_BUDGET_EXHAUSTED')
    log = OUT / (SCRATCH.name+'-'+label+'.log')
    begin = time.monotonic()
    with log.open('wb') as f:
        p = subprocess.Popen(argv, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ,'LEAN_NUM_THREADS':'1'}, start_new_session=True)
        try:
            p.wait(timeout=left)
        except BaseException:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); raise
    text = log.read_text(errors='replace')
    RECORDS.append(dict(label=label,argv=argv,exit=p.returncode,seconds=time.monotonic()-begin,log=str(log)))
    if require and p.returncode:
        raise RuntimeError(f'{label}_REJECT {p.returncode}\n{text[-3500:]}')
    return p.returncode,text

# The real source closure the wrapper compiles: `src/engine/ops/moe.rs` imports
# `super::{ffn,matvec,value,ONE}` and `crate::numeric::fixed_exp::fixed_exp`; `ffn.rs` imports
# `super::{matvec,par,value}`; `par.rs` imports `super::{fused,MatvecError}`; `fused.rs` imports
# `super::{dequant,matvec,value}`; `matvec.rs` imports `crate::format::quant` and
# `super::{dequant,value,EngineError}`; `error.rs` imports `crate::engine::f32_convert` and
# `crate::format::{gguf,quant}`. Modules are declared at the crate root (a `#[path]` inside an
# inline module resolves against a virtual directory) with `format`/`engine`/`numeric` mirrors and
# the crate-root aliases, exactly the wiring `proof/extract/lib.rs` uses; the aliases supply the
# `super::ONE`, `super::EngineError`, `super::super::{matvec_fixed,EngineError}` and
# `crate::engine::*` names the real module tests read, so `cargo test --lib` runs the production
# `moe.rs` tests against the extracted body.
LIB_TEMPLATE = '''pub mod format {{
    pub use super::{{gguf, quant}};
}}
pub mod numeric {{
    pub use super::fixed_exp;
}}
pub mod engine {{
    pub use super::{{dequant, error, f32_convert, ffn, matvec, moe, par, value}};
    pub use super::dequant::f16_to_fixed;
    pub use super::error::EngineError;
    pub use super::matvec::{{MatvecError, matvec_fixed}};
    pub use super::moe::MoeError;
    pub const FIXED_FRAC_BITS: u32 = 32;
    pub const ONE: i64 = 1 << FIXED_FRAC_BITS;
}}
#[path = {quant}]
pub mod quant;
#[path = {gguf}]
pub mod gguf;
#[path = {fixed_exp}]
pub mod fixed_exp;
#[path = {value}]
pub mod value;
#[path = {f32_convert}]
pub mod f32_convert;
#[path = {error}]
pub mod error;
#[path = {dequant}]
pub mod dequant;
#[path = {matvec}]
pub mod matvec;
#[path = {par}]
pub mod par;
#[path = {fused}]
pub mod fused;
#[path = {ffn}]
pub mod ffn;
#[path = {moe}]
pub mod moe;
pub use engine::{{FIXED_FRAC_BITS, ONE}};
pub use error::EngineError;
pub use matvec::{{MatvecError, matvec_fixed}};

/// Typed forwarding wrapper: it fixes the slice, width and result types and calls the real
/// `src/engine/ops/moe.rs` `route_topk` body, exactly as `proof/extract/lib.rs`
/// `word_route_topk` does. It restates no formula.
pub fn word_route_topk(
    logits: &[i64],
    k: usize,
) -> Result<Vec<u32>, engine::moe::MoeError> {{
    engine::moe::route_topk(logits, k)
}}
'''

def lib_source(moe):
    """Absolute `#[path]`s to the real modules; only `moe` varies with the control copy."""
    src = ROOT/'src'
    def p(rel):
        return json.dumps(str(src/rel))
    return LIB_TEMPLATE.format(quant=p('format/quant.rs'), gguf=p('format/gguf.rs'),
        fixed_exp=p('numeric/fixed_exp.rs'), value=p('engine/value.rs'),
        f32_convert=p('engine/f32_convert.rs'), error=p('engine/error.rs'),
        dequant=p('engine/dequant.rs'), matvec=p('engine/matvec.rs'), par=p('engine/par.rs'),
        fused=p('engine/fused.rs'), ffn=p('engine/ffn.rs'), moe=json.dumps(str(moe)))

def extract(source, label):
    """Create only a typed forwarding wrapper and extract its real call closure."""
    project = SCRATCH/label
    project.mkdir()
    (project/'Cargo.toml').write_text('[package]\nname="route_topk_source"\nversion="0.0.0"\nedition="2021"\n[lib]\npath="lib.rs"\n[target.\'cfg(target_os = "linux")\'.dependencies]\nlibc = "=0.2.189"\n[workspace]\n')
    (project/'lib.rs').write_text(lib_source(source))
    run(['cargo','generate-lockfile','--offline'], project,label+'-lock')
    run(['cargo','test','--locked','--offline','--lib','--target-dir',str(SCRATCH/'rust-target')],project,label+'-rust-tests',require=(label != 'mutant'))
    llbc = project/'route_topk_source.llbc'
    run([str(TOOLS/'target/aeneas/charon'),'cargo','--sysroot','default','--preset','aeneas','--dest-file',str(llbc),
         '--start-from','crate::word_route_topk','--','--manifest-path',str(project/'Cargo.toml'),'--locked','--offline','--lib',
         '--target-dir',str(SCRATCH/'charon-target')],project,label+'-charon')
    run([str(TOOLS/'target/aeneas/aeneas'),'-backend','lean','-dest',str(project),'-warnings-as-errors',
         '-no-progress-bar','-all-computable',str(llbc)],project,label+'-aeneas')
    generated=project/'RouteTopkSource.lean'
    # Aeneas may leave blanks at line ends; normalize as proof/regen.sh does.
    generated.write_text('\n'.join(line.rstrip() for line in generated.read_text().split('\n')))
    # The model's own definitions replace the stubs the pinned Aeneas emits for externals it does
    # not model; any other axiom/sorry stays for the fail-closed scan below (proof/regen.sh does
    # exactly this, with the same repository-owned allowlist).
    run(['python3',str(ROOT/'proof/strip-modeled-axioms.py'),str(generated)],project,label+'-strip')
    generated.write_text(generated.read_text().replace(str(source), 'src/engine/ops/moe.rs'))
    if re.search(r'\b(axiom|sorry|admit)\b', generated.read_text()):
        raise RuntimeError('ROUTE_TOPK_UNMODELED_SOURCE')
    return generated

def prepare_lean():
    """Private local package; dependency artifacts read from the pinned compatible lane."""
    p=SCRATCH/'lean';p.mkdir()
    (p/'lean-toolchain').write_bytes((TOOLS/'proof/dispatch/lean-toolchain').read_bytes())
    (p/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage routeTopkProof where\n  leanOptions := #[⟨`autoImplicit,false⟩,⟨`warningAsError,true⟩]\nrequire aeneas from '+json.dumps(str(TOOLS/'target/aeneas-model/lean'))+'\n@[default_target] lean_lib RouteTopk where\n  roots := #[`RouteTopkSource,`LoopTrace,`RouteTopkResource]\n')
    manifest=json.loads((TOOLS/'proof/dispatch/lake-manifest.json').read_text());manifest['name']='routeTopkProof'
    for dep in manifest['packages']:
        if dep['name']=='aeneas': dep['dir']=str(TOOLS/'target/aeneas-model/lean')
    (p/'lake-manifest.json').write_text(json.dumps(manifest))
    deps=p/'.lake/packages';deps.mkdir(parents=True)
    for dep in (TOOLS/'proof/dispatch/.lake/packages').iterdir(): (deps/dep.name).symlink_to(dep.resolve(),target_is_directory=True)
    return p

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--check',action='store_true');args.add_argument('--controls',action='store_true');a=args.parse_args()
    source=ROOT/'src/engine/ops/moe.rs'
    run(['rustc','--version'],ROOT,'rustc-version')
    run(['cargo','--version'],ROOT,'cargo-version')
    run(['lean','--version'],ROOT,'lean-version')
    generated=extract(source,'fresh')
    dest=ROOT/'proof/resource/RouteTopkSource.lean'
    if a.prepare:
        shutil.copyfile(generated,dest);print('ROUTE_TOPK_EXTRACTION_READY');return
    if generated.read_bytes()!=dest.read_bytes():raise RuntimeError('ROUTE_TOPK_EXTRACTION_DRIFT')
    proof=ROOT/'proof/resource/RouteTopkResource.lean'
    if re.search(r'\b(sorry|admit|axiom)\b|set_option maxHeartbeats 0',proof.read_text()):raise RuntimeError('ROUTE_TOPK_PROOF_POLICY_REJECT')
    p=prepare_lean();shutil.copyfile(dest,p/'RouteTopkSource.lean');shutil.copyfile(LOOPTRACE,p/'LoopTrace.lean');shutil.copyfile(proof,p/'RouteTopkResource.lean')
    code,text=run(['lake','build','RouteTopkResource'],p,'fresh-proof')
    if 'ROUTE_TOPK_KERNEL_OK' not in text:raise RuntimeError('ROUTE_TOPK_AUDIT_MISSING')
    if a.controls:
        # Real-source mutation: the selection loop's scan cursor advance, `pass += 1;` at
        # src/engine/ops/moe.rs:245, becomes `pass += 2;`. A resource claim about the k passes and
        # their E-long scans must see it; a scan-count claim that does not see it has no weight.
        original=source.read_text();needle='        pass += 1;'
        if original.count(needle)!=1:raise RuntimeError('ROUTE_TOPK_CONTROL_TARGET_MISSING')
        mutant=SCRATCH/'moe.rs';mutant.write_text(original.replace(needle,'        pass += 2;'))
        freshmut=extract(mutant,'mutant');shutil.copyfile(freshmut,p/'RouteTopkSource.lean')
        code,text=run(['lake','build','RouteTopkResource'],p,'mutant-proof',require=False)
        if code!=1 or 'RouteTopkResource.lean:' not in text or not any(x in text for x in ['unsolved goals','type mismatch','Tactic']):
            raise RuntimeError('ROUTE_TOPK_CONTROL_NOT_ATTRIBUTABLE '+text[-1800:])
        print('ROUTE_TOPK_REAL_SOURCE_CONTROL_RED')
        mutant.write_text(original);restored=extract(mutant,'restored')
        # Source-location comments differ for disposable absolute source paths; compare executable text.
        strip=lambda s: re.sub(r'/\*.*?\*/|/--.*?-/', '',s,flags=re.S)
        if strip(restored.read_text())!=strip(dest.read_text()):raise RuntimeError('ROUTE_TOPK_RESTORE_BODY_DRIFT')
        shutil.copyfile(dest,p/'RouteTopkSource.lean');run(['lake','build','RouteTopkResource'],p,'restored-proof')
        print('ROUTE_TOPK_REAL_SOURCE_CONTROL_RESTORED')
    print('ROUTE_TOPK_SOURCE_RESOURCE_OK')

if __name__=='__main__':
    status=1
    try:
        main();status=0
    finally:
        source_paths=[TOOLS/'target/aeneas/charon-driver',TOOLS/'target/aeneas-model/lean/lean-toolchain',TOOLS/'proof/dispatch/lean-toolchain',ROOT/'src/engine/ops/moe.rs',ROOT/'proof/strip-modeled-axioms.py',ROOT/'proof/resource/RouteTopk_gate.py',ROOT/'proof/resource/RouteTopkSource.lean',ROOT/'proof/resource/RouteTopkResource.lean',ROOT/'proof/native/LoopTrace.lean',TOOLS/'target/aeneas/charon',TOOLS/'target/aeneas/aeneas',TOOLS/'proof/dispatch/lake-manifest.json']
        (OUT/(SCRATCH.name+'-receipt.json')).write_text(json.dumps(dict(exit=status,seconds=time.monotonic()-START,budget=540,argv=[sys.executable,*sys.argv],lane=LANE,memory_max='12G',swap_max=0,threads=1,scratch=str(SCRATCH),deadline=os.environ.get('ROUTE_TOPK_DEADLINE'),records=RECORDS,sha256={str(p):digest(p) for p in source_paths if p.is_file()},trust='Aeneas/Charon, compiler, IO/process/dependencies; scalar comparison/scan-count claim excludes GPU/hardware timing'),indent=2)+'\n')
        shutil.rmtree(SCRATCH,ignore_errors=True)
