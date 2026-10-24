//! `NanoQuant`'s sequential block loop on the device: `compress_block_recon`
//! (`SamsungLabs/NanoQuant` @ `a9e0a43`, `src/nanoquant/core/compress_model.py:19-120`) over a
//! block's seven linears in the reference's order `q, v, o, k, gate, up, down`
//! (`src/nanoquant/utils/utils.py:77-88`).
//!
//! Per block `i` (`compress_model.py:56-113`): the teacher's output on the fp stream
//! (`:58-63`), then per linear `tune_nonfact` (`core/compress_block.py:52-98`, lr 1e-4, batch 4,
//! 8 epochs, `CosineAnnealingLR(T_max=256, eta_min=1e-8)`, one `randperm(128)` per epoch after
//! `set_seed(0)`), ADMM (`compress_model.py:107-153`, our `learn::nanoquant::factorize` on the
//! nonfact-tuned `W`), and `tune_fact` (`core/compress_block.py:160-201`, lr 1e-5, batch 1,
//! 8 epochs, `T_max=1024`, `eta_min=1e-9`), then `finalize` (`modules/linear.py:264-283`). The
//! loss is `((y − y_t)²·imp).sum()` (`compress_block.py:17-19`) with `imp` the block's
//! `mlp.down_proj.o_norm` (`compress_model.py:67-78`); its gradient is
//! `(loss/batch_size).backward()`, so the device step's loss scale is `2/bs <
//! [BlockTune::set_loss_scale]>`. The optimiser is optimi's `AdamW` (0.9, 0.99), eps 1e-6, wd 0
//! (`AdamHyper::OPTIMI`).
//!
//! The two device streams are the fp teacher's input `xt` and the student's `xs`, both held as an
//! f32 carrier of bf16-rounded values (`utils/load_utils.py:126,164`); `yt` is the teacher's block
//! output and `next` the student's. The device tapes are `gpu::nn::train::front` (the attention
//! front, `q`/`k`/`v` as factors) and `gpu::nn::train::block` (the o-projection and the `SwiGLU`
//! FFN), run as one whole-block step: front forward → `ao` into the block's `a` → block taper →
//! `da` into the front's `dao` → front update.
//!
//! # Stated deviations from the reference
//!
//! - **Precision.** The reference's parameters and streams are bf16 with Kahan summation in the
//!   optimiser; ours are f32 without (`AdamHyper::OPTIMI`'s documented `ε` deviation). Optimi holds
//!   the bf16 parameter's moments in bf16 too (`optimi/adam.py:99-101`). Measured at block 0, q/v:
//!   v nonfact 13.5–17.9% below the reference every epoch; q fact within 3.9%, v fact within 0.3%.
//! - **Preconditioner floor.** The reference clamps `sqrt(diag)` at its own `eps = 1e-12`; ADMM here
//!   runs `factorize_floor` with floor `APEIRON_NQ_FLOOR` (default `0`, the reference's clamp).
//!
//! # Knobs
//!
//! `APEIRON_NQ_BLOCKS=a..b` (blocks this round), `APEIRON_NQ_SLOTS=q,v` (a subset of the order),
//! `APEIRON_NQ_PHASES=n` (phases this round), `APEIRON_NQ_STREAM=f32` (no bf16 rounding),
//! `APEIRON_NQ_ADMM=a` (inject the dump's block-0 factor instead of fitting), `APEIRON_NQ_ARM=b0`
//! (the host-only ADMM arm). Checkpoints under `~/nq-ref/loop/<tag>/`.
#![cfg(feature = "gpu")]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    reason = "test geometry: single letters are the tape's symbols; token ids, widths and workgroup \
              counts index u32 dispatches, and f64 statistics narrow to the device's f32"
)]

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::gpu::decode::GpuCtx;
use apeiron::gpu::nn::train::block::{
    chunk_from, chunk_into, stream_upload, BlockShape, BlockTune, HostFactor, Stream,
};
use apeiron::gpu::nn::train::front::{FrontShape, FrontTune};
use apeiron::gpu::nn::train::mixer::{LayerFwd, LayerHost, Mixer};
use apeiron::gpu::nn::{cosine_lr, AdamHyper};
use apeiron::gpu::vulkan::{
    check, BufferCopy, CommandBuffer, Handle, Vk, ACCESS_HOST_READ, ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE, ACCESS_TRANSFER_READ, ACCESS_TRANSFER_WRITE, BUFFER_USAGE_TRANSFER_DST,
    BUFFER_USAGE_TRANSFER_SRC, MEMORY_HOST_COHERENT, MEMORY_HOST_VISIBLE, STAGE_COMPUTE,
    STAGE_HOST, STAGE_TRANSFER,
};
use apeiron::learn::grad::load::load_rows;
use apeiron::learn::grad::model::Config;
use apeiron::learn::nanoquant::{factorize_floor, NqConfig, NqFactor};

/// The teacher, `Qwen3-0.6B-Base` in f16 (`APEIRON_NQ_MODEL`).
const MODEL: &str = "~/nq-ref/q3-0p6b-f16.gguf";
/// Part A's dump: `calib_ids.i32`, `perm.json`, `ranks.json`, `stats/`, `b0/` (`APEIRON_NQ_DUMP`).
const DUMP: &str = "~/nq-ref/dump";
/// Checkpoint root (`APEIRON_NQ_LOOP`); `<root>/<tag>/`.
const LOOP: &str = "~/nq-ref/loop";
/// The reference's block-0 input and target for sample 0, for the two stream checks.
const B0: &str = "b0";
/// Columns per device column-step: the four samples of a `tune_nonfact` batch
/// (`quant_config['nonfact_batch_size']`, `main.py:69`), `2048` tokens each.
const PER_STEP: usize = 4;
/// `attn_map`'s score buffer, `src/gpu/spirv/train/attn_map.comp:37`: the longest sequence one
/// device dispatch can attend over.
const N_CTX: usize = 2048;
/// The base rate the two tuners are built with; each phase's rate arrives as the schedule
/// multiplier `c = lr_t / base` (`BlockTune::set_schedule`, `AdamHyper`'s `adam_scheduled_bias`).
const LR_BASE: f64 = 1e-5;
/// The reference's seven linears in its own order (`utils/utils.py:77-88`).
const SLOT_KEYS: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.k_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];
/// Each slot's index in `FrontTune`'s `[q, k, v]`.
const SLOT_FRONT: [Option<usize>; 7] = [Some(0), Some(2), None, Some(1), None, None, None];
/// Each slot's index in `BlockTune`'s `[o, gate, up, down]`.
const SLOT_BLOCK: [Option<usize>; 7] = [None, None, Some(0), None, Some(1), Some(2), Some(3)];

fn env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}

fn envu(key: &str, default: usize) -> usize {
    std::env::var(key).map_or(default, |v| v.parse().expect(key))
}

fn envf(key: &str, default: f64) -> f64 {
    std::env::var(key).map_or(default, |v| v.parse().expect(key))
}

/// A path under the dump.
fn dump_at(parts: &str) -> String {
    format!("{DUMP}/{parts}")
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4")))
        .collect()
}

fn read_i32(path: &str) -> Vec<i32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().expect("4")))
        .collect()
}

/// The dump's `bf16` payloads: the high half of an f32.
fn read_bf16(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

/// An f32 rounded to bf16 and back: `torch.Tensor.bfloat16()`'s round-to-nearest-even.
fn round_bf16(x: f32) -> f32 {
    let b = x.to_bits();
    if (b & 0x7fff) == 0x7fff {
        return x;
    }
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// `ranks.json`'s `{"<layer>.<key>": rank}` as `[layer][slot]`, slots in [`SLOT_KEYS`] order.
fn ranks(dump: &str) -> Vec<[usize; 7]> {
    let s = std::fs::read_to_string(format!("{dump}/ranks.json")).expect("ranks.json");
    let mut out: Vec<[usize; 7]> = Vec::new();
    for piece in s.split(',') {
        let Some(q) = piece.find('"') else { continue };
        let Some(q2) = piece[q + 1..].find('"') else {
            continue;
        };
        let key = &piece[q + 1..q + 1 + q2];
        let (layer, name) = key.split_once('.').expect("layer.slot key");
        let Some(num) = piece.rsplit(':').next() else {
            continue;
        };
        let rank: usize = num
            .trim()
            .trim_end_matches(['}', ']', ' ', '\n'])
            .parse()
            .unwrap_or_else(|e| panic!("{piece}: {e}"));
        let l: usize = layer.parse().expect("layer");
        let k = SLOT_KEYS
            .iter()
            .position(|x| *x == name)
            .unwrap_or_else(|| panic!("rank key {key}"));
        if out.len() <= l {
            out.resize(l + 1, [0; 7]);
        }
        out[l][k] = rank;
    }
    assert!(!out.is_empty(), "ranks.json: no rows");
    out
}

/// `perm.json`'s draws in call order (`torch.randperm(128)` under `set_seed(0)`), each with the
/// calling function's name.
fn perms(dump: &str) -> Vec<(String, Vec<usize>)> {
    let s = std::fs::read_to_string(format!("{dump}/perm.json")).expect("perm.json");
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(i) = s[at..].find("\"index\"") {
        let begin = at + i;
        let Some(open) = s[begin..].find('[') else {
            break;
        };
        let Some(close) = s[begin + open..].find(']') else {
            break;
        };
        let body = &s[begin + open + 1..begin + open + close];
        let idx: Vec<usize> = body
            .split(',')
            .map(|t| t.trim().parse().expect("perm entry"))
            .collect();
        let caller = s[..begin]
            .rsplit("\"caller\"")
            .next()
            .and_then(|t| t.split('"').nth(2))
            .unwrap_or("")
            .to_owned();
        out.push((caller, idx));
        at = begin + open + close;
    }
    assert!(!out.is_empty(), "perm.json: no draws");
    out
}

/// A host-visible, host-coherent f32 buffer and its mapped pointer: the staging and readback
/// window every transfer below goes through.
struct Stage {
    buf: Handle,
    ptr: *mut f32,
    n: usize,
}

unsafe fn stage_new(g: &GpuCtx, n: usize, usage: u32) -> Stage {
    let (buf, mem) =
        g.vk.buffer(
            n.max(1) as u64 * 4,
            usage,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
        .expect("staging buffer");
    Stage {
        buf,
        ptr: g.vk.map(mem).cast::<f32>(),
        n,
    }
}

/// Copy `data` into `dst` at word offset `at`, through `st`.
unsafe fn put_at(g: &GpuCtx, cb: CommandBuffer, st: &Stage, dst: Handle, at: usize, data: &[f32]) {
    assert!(data.len() <= st.n, "put_at: {} words", data.len());
    std::ptr::copy_nonoverlapping(data.as_ptr(), st.ptr, data.len());
    let vk = g.vk;
    vk.reset_cb(cb);
    vk.begin(cb);
    let region = BufferCopy {
        src_offset: 0,
        dst_offset: (at * 4) as u64,
        size: (data.len() * 4) as u64,
    };
    (vk.copy_buffer)(cb, st.buf, dst, 1, &raw const region);
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_COMPUTE,
        ACCESS_TRANSFER_WRITE,
        ACCESS_SHADER_READ,
    );
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
}

/// Read `n` words of `src` from word offset `at` through `st`.
unsafe fn get_at(
    g: &GpuCtx,
    cb: CommandBuffer,
    st: &Stage,
    src: Handle,
    at: usize,
    n: usize,
) -> Vec<f32> {
    assert!(n <= st.n, "get_at: {n} words");
    let vk = g.vk;
    vk.reset_cb(cb);
    vk.begin(cb);
    vk.barrier(
        cb,
        STAGE_COMPUTE,
        STAGE_TRANSFER,
        ACCESS_SHADER_WRITE,
        ACCESS_TRANSFER_READ,
    );
    let region = BufferCopy {
        src_offset: (at * 4) as u64,
        dst_offset: 0,
        size: (n * 4) as u64,
    };
    (vk.copy_buffer)(cb, src, st.buf, 1, &raw const region);
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_HOST,
        ACCESS_TRANSFER_WRITE,
        ACCESS_HOST_READ,
    );
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
    std::slice::from_raw_parts(st.ptr, n).to_vec()
}

/// One submission of device-to-device copies: `(src, src word, dst, dst word, words)` each.
unsafe fn submit_copies(
    g: &GpuCtx,
    cb: CommandBuffer,
    groups: &[(Handle, usize, Handle, usize, usize)],
) {
    if groups.is_empty() {
        return;
    }
    let vk = g.vk;
    vk.reset_cb(cb);
    vk.begin(cb);
    vk.barrier(
        cb,
        STAGE_COMPUTE,
        STAGE_TRANSFER,
        ACCESS_SHADER_WRITE,
        ACCESS_TRANSFER_READ,
    );
    for &(src, sa, dst, da, n) in groups {
        let region = BufferCopy {
            src_offset: (sa * 4) as u64,
            dst_offset: (da * 4) as u64,
            size: (n * 4) as u64,
        };
        (vk.copy_buffer)(cb, src, dst, 1, &raw const region);
    }
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_COMPUTE,
        ACCESS_TRANSFER_WRITE,
        ACCESS_SHADER_READ,
    );
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
}

/// Round the `ids` regions of `s` to bf16 in place, in `st.n`-word slices.
unsafe fn round_ids(g: &GpuCtx, cb: CommandBuffer, st: &Stage, s: &Stream, ids: &[usize]) {
    let c = s.chunk_words();
    for &id in ids {
        let mut at = 0;
        while at < c {
            let take = st.n.min(c - at);
            let mut v = get_at(g, cb, st, s.buf, id * c + at, take);
            for x in &mut v {
                *x = round_bf16(*x);
            }
            put_at(g, cb, st, s.buf, id * c + at, &v);
            at += take;
        }
    }
}

/// Write the whole of `s` to `path` as raw little-endian f32.
unsafe fn save_stream(g: &GpuCtx, cb: CommandBuffer, st: &Stage, s: &Stream, path: &str) {
    let tmp = format!("{path}.tmp");
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(&tmp).unwrap_or_else(|e| panic!("{tmp}: {e}")),
    );
    let n = s.words();
    let mut at = 0;
    while at < n {
        let take = st.n.min(n - at);
        for x in get_at(g, cb, st, s.buf, at, take) {
            f.write_all(&x.to_le_bytes()).expect("write");
        }
        at += take;
    }
    f.flush().expect("flush");
    drop(f);
    std::fs::rename(&tmp, path).expect("rename");
}

/// Read `path` (raw little-endian f32) into the whole of `s`.
unsafe fn load_stream(g: &GpuCtx, cb: CommandBuffer, st: &Stage, s: &Stream, path: &str) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(bytes.len(), s.words() * 4, "{path}: stream words");
    let v: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4")))
        .collect();
    let mut at = 0;
    while at < s.words() {
        let take = st.n.min(s.words() - at);
        put_at(g, cb, st, s.buf, at, &v[at..at + take]);
        at += take;
    }
}

/// The shrunk statistic `which` (`i_norm` / `o_norm`) of linear `key` of block `l`
/// (`compress_model.py:69-78`, `load_utils.py:126,164`).
fn stat(dump: &str, l: usize, key: &str, which: &str) -> Vec<f32> {
    read_f32(&format!("{dump}/stats/model.layers.{l}.{key}.{which}.f32"))
}

/// The factor `factorize_admm_nanoquant` returned for block `l`'s linear `key`, as the tape's
/// host image: `W = diag(s_post)·sign(U)·sign(V)ᵀ·diag(s_pre)` with `U = A_latentᵀ` (`o × r`) and
/// `V = B_latent` (`r × n`) (`core/admm_nq.py:215-250`, `modules/linear.py:44-56`).
fn dump_factor(dump: &str, key: &str) -> HostFactor {
    let a = read_f32(&format!("{dump}/{B0}/{key}.admm.A_latent.f32"));
    let b = read_f32(&format!("{dump}/{B0}/{key}.admm.B_latent.f32"));
    let s_pre = read_f32(&format!("{dump}/{B0}/{key}.admm.scale_pre.f32"));
    let s_post = read_f32(&format!("{dump}/{B0}/{key}.admm.scale_post.f32"));
    let o = s_post.len();
    let n = s_pre.len();
    let r = b.len() / n;
    assert_eq!((a.len(), b.len()), (r * o, r * n), "{key}: ADMM shapes");
    let mut u = vec![0f32; o * r];
    let mut vt = vec![0f32; n * r];
    for k in 0..r {
        for i in 0..o {
            u[i * r + k] = a[k * o + i];
        }
        for c in 0..n {
            vt[c * r + k] = b[k * n + c];
        }
    }
    HostFactor {
        out: o,
        inn: n,
        rank: r,
        u,
        vt,
        s_post,
        s_pre,
    }
}

/// The tape's host image of a fitted factor: the latents as they are, `V` transposed to `n × r`.
fn host_from_nq(f: &NqFactor) -> (HostFactor, usize) {
    let (o, n, r) = (f.out, f.inn, f.rank);
    let mut flips = 0;
    for i in 0..o * r {
        if (f.u_latent[i] < 0.0) != (f.u[i] < 0) {
            flips += 1;
        }
    }
    let mut vt = vec![0f32; n * r];
    for k in 0..r {
        for c in 0..n {
            vt[c * r + k] = f.v_latent[k * n + c] as f32;
            if (f.v_latent[k * n + c] < 0.0) != (f.v[k * n + c] < 0) {
                flips += 1;
            }
        }
    }
    let h = HostFactor {
        out: o,
        inn: n,
        rank: r,
        u: f.u_latent.iter().map(|&v| v as f32).collect(),
        vt,
        s_post: f.s_post.iter().map(|&v| v as f32).collect(),
        s_pre: f.s_pre.iter().map(|&v| v as f32).collect(),
    };
    (h, flips)
}

/// A factor as an [`NqFactor`]: the signs of the latents (`sign(0) = +1`, `linear.py:120-123`) and
/// the latents themselves.
fn nq_from_host(h: &HostFactor, admm_loss: f64) -> NqFactor {
    let (o, n, r) = (h.out, h.inn, h.rank);
    let sg = |v: f32| if v < 0.0 { -1i8 } else { 1 };
    let mut v = vec![0i8; r * n];
    let mut v_latent = vec![0f64; r * n];
    for k in 0..r {
        for c in 0..n {
            v[k * n + c] = sg(h.vt[c * r + k]);
            v_latent[k * n + c] = f64::from(h.vt[c * r + k]);
        }
    }
    NqFactor {
        out: o,
        inn: n,
        rank: r,
        u: h.u.iter().map(|&x| sg(x)).collect(),
        v,
        s_pre: h.s_pre.iter().map(|&x| f64::from(x)).collect(),
        s_post: h.s_post.iter().map(|&x| f64::from(x)).collect(),
        admm_loss,
        u_latent: h.u.iter().map(|&x| f64::from(x)).collect(),
        v_latent,
    }
}

/// The block's seven linears as the tapes hold them: which are factors, each slot's dense `W`, and
/// one factor image per slot (only read for a factored slot; a dense slot's is its untouched
/// latent buffer, which the tape skips, `block.rs:613-619`).
#[derive(Clone)]
struct Params {
    fd: [bool; 3],
    bd: [bool; 4],
    fdims: [(usize, usize); 3],
    bdims: [(usize, usize); 4],
    fw: [Vec<f32>; 3],
    bw: [Vec<f32>; 4],
    ff: [HostFactor; 3],
    bf: [HostFactor; 4],
}

/// An all-ones factor of the given geometry: the placeholder a dense slot carries.
fn ones_factor(out: usize, inn: usize, rank: usize) -> HostFactor {
    HostFactor {
        out,
        inn,
        rank,
        u: vec![1.0; out * rank],
        vt: vec![1.0; inn * rank],
        s_post: vec![1.0; out],
        s_pre: vec![1.0; inn],
    }
}

/// The teacher's dense block, every slot still a `nn.Linear`.
fn teacher_params(host: &LayerHost, fs: FrontShape, bs: BlockShape) -> Params {
    let mut ff = [(); 3].map(|()| ones_factor(0, 0, 0));
    let mut bf = [(); 4].map(|()| ones_factor(0, 0, 0));
    for k in 0..3 {
        let (o, i) = fs.dims(k);
        ff[k] = ones_factor(o, i, fs.ranks[k]);
    }
    for k in 0..4 {
        let (o, i) = bs.dims(k);
        bf[k] = ones_factor(o, i, bs.ranks[k]);
    }
    Params {
        fd: [false; 3],
        bd: [false; 4],
        fdims: [fs.dims(0), fs.dims(1), fs.dims(2)],
        bdims: [bs.dims(0), bs.dims(1), bs.dims(2), bs.dims(3)],
        fw: [
            host.proj[0].1.clone(),
            host.proj[1].1.clone(),
            host.proj[2].1.clone(),
        ],
        bw: [
            host.wo.clone(),
            host.ffn[0].clone(),
            host.ffn[1].clone(),
            host.ffn[2].clone(),
        ],
        ff,
        bf,
    }
}

/// Read both tuners' parameters out: `O(Σ out·inn + Σ (out + inn)·rank)` words downloaded.
unsafe fn read_params(g: &GpuCtx, p: &mut Params, bt: &BlockTune, ft: &FrontTune) {
    let ffs = ft.factors(g);
    for k in 0..3 {
        if p.fd[k] {
            p.ff[k] = ffs[k].clone();
        } else {
            p.fw[k] = ft.weight(g, k);
        }
    }
    let bfs = bt.factors(g);
    for k in 0..4 {
        if p.bd[k] {
            p.bf[k] = bfs[k].clone();
        } else {
            p.bw[k] = bt.weight(g, k);
        }
    }
}

/// Put `p` into both tuners and restart every Adam counter: the reference builds a fresh optimiser
/// for each `tune_nonfact` / `tune_fact` (`compress_block.py:73-74`, `:182-185`), so every moment
/// starts at zero and every step number at one.
unsafe fn engage(
    g: &GpuCtx,
    bt: &mut BlockTune,
    ft: &mut FrontTune,
    p: &Params,
    gains: (&[f32], &[f32], &[f32]),
    sched: &[f64],
    scale: f32,
    train: ([bool; 3], [bool; 4]),
) {
    ft.load(g, &p.ff, gains.0, gains.1);
    for k in 0..3 {
        if !p.fd[k] {
            ft.load_dense(g, k, &p.fw[k]);
        }
    }
    ft.set_stage(g, p.fd, train.0);
    ft.set_schedule(sched.to_vec());
    bt.load(g, &p.bf, gains.2);
    for k in 0..4 {
        if !p.bd[k] {
            bt.load_dense(g, k, &p.bw[k]);
        }
    }
    bt.set_stage(g, p.bd, train.1);
    bt.set_schedule(sched.to_vec());
    bt.set_loss_scale(g, scale);
}

/// The streams, the tuners' shared buffers and the two transfer windows one step reads.
struct Bund<'a, 'b> {
    g: &'a GpuCtx<'b>,
    cb: CommandBuffer,
    st: &'a Stage,
    xs: &'a Stream,
    yt: &'a Stream,
    next: &'a Stream,
    t: usize,
    a: usize,
    bf16: bool,
}

/// One whole-block step over `ids`: the front's forward, `ao` into the block's `a`, the block's
/// forward/backward/update, and (when a front slot is in the trained set, `front_train`) the
/// block's `da` into the front's `dao` and the front's own backward and update. Returns the step's
/// weighted sum of squares, before the update (`compress_block.py:88-96`).
unsafe fn step(
    b: &Bund,
    bt: &mut BlockTune,
    ft: &mut FrontTune,
    ids: &[usize],
    front_train: bool,
) -> f64 {
    let (bx, ba, byt) = bt.cols();
    let (fx, _) = ft.cols();
    let c = b.xs.chunk_words();
    let pre: Vec<(Handle, usize, Handle, usize, usize)> = ids
        .iter()
        .enumerate()
        .map(|(k, &id)| (b.xs.buf, id * c, fx, k * c, c))
        .collect();
    submit_copies(b.g, b.cb, &pre);
    ft.forward(b.g);
    submit_copies(b.g, b.cb, &[(ft.ao(), 0, ba, 0, b.t * b.a)]);
    let pre = vec![chunk_into(b.xs, ids, bx), chunk_into(b.yt, ids, byt)];
    bt.set_cols(b.g, pre, Vec::new(), true);
    let loss = bt.run(b.g, true);
    if front_train {
        let (_, dao) = ft.cols();
        submit_copies(b.g, b.cb, &[(bt.da(), 0, dao, 0, b.t * b.a)]);
        ft.train(b.g);
    }
    loss
}

/// One evaluation pass over `ids` on the student stream: the front's forward, then the block's
/// evaluation tape, leaving the block output in `bt.out()`.
unsafe fn eval_pass(b: &Bund, bt: &mut BlockTune, ft: &mut FrontTune, ids: &[usize]) {
    let (bx, ba, byt) = bt.cols();
    let (fx, _) = ft.cols();
    let c = b.xs.chunk_words();
    let pre: Vec<(Handle, usize, Handle, usize, usize)> = ids
        .iter()
        .enumerate()
        .map(|(k, &id)| (b.xs.buf, id * c, fx, k * c, c))
        .collect();
    submit_copies(b.g, b.cb, &pre);
    ft.forward(b.g);
    submit_copies(b.g, b.cb, &[(ft.ao(), 0, ba, 0, b.t * b.a)]);
    let pre = vec![chunk_into(b.xs, ids, bx), chunk_into(b.yt, ids, byt)];
    bt.set_cols(b.g, pre, vec![chunk_from(b.next, ids, bt.out())], false);
    bt.run(b.g, false);
    if b.bf16 {
        round_ids(b.g, b.cb, b.st, b.next, ids);
    }
}

/// The reference's linear names in `FrontTune`'s `[q, k, v]` order.
const FRONT_KEYS: [&str; 3] = [SLOT_KEYS[0], SLOT_KEYS[3], SLOT_KEYS[1]];
/// The reference's linear names in `BlockTune`'s `[o, gate, up, down]` order.
const BLOCK_KEYS: [&str; 4] = [SLOT_KEYS[2], SLOT_KEYS[4], SLOT_KEYS[5], SLOT_KEYS[6]];

/// A checkpointed parameter: a dense `W` or a factor.
enum Param {
    Dense { out: usize, inn: usize, w: Vec<f32> },
    Factor(HostFactor),
}

/// `[out, inn, rank, kind]`, then the payload: a dense `W` (`kind` 0) or a factor's `u`, `vt`,
/// `s_post`, `s_pre` (`kind` 1).
fn save_param(path: &str, p: &Param) {
    let mut b = Vec::new();
    let (h, mut vals): ([u64; 4], Vec<&[f32]>) = match p {
        Param::Dense { out, inn, w } => ([*out as u64, *inn as u64, 0, 0], vec![w]),
        Param::Factor(x) => (
            [x.out as u64, x.inn as u64, x.rank as u64, 1],
            vec![&x.u, &x.vt, &x.s_post, &x.s_pre],
        ),
    };
    for x in h {
        b.extend_from_slice(&x.to_le_bytes());
    }
    for v in vals.drain(..) {
        for &x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, &b).unwrap_or_else(|e| panic!("{tmp}: {e}"));
    std::fs::rename(&tmp, path).expect("rename");
}

fn load_param(path: &str) -> Param {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let word = |i: usize| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().expect("word"));
    let (out, inn, rank) = (word(0) as usize, word(1) as usize, word(2) as usize);
    let mut at = 32;
    let mut take = |n: usize| -> Vec<f32> {
        let v = b[at..at + 4 * n]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("f32")))
            .collect();
        at += 4 * n;
        v
    };
    if word(3) == 0 {
        return Param::Dense {
            out,
            inn,
            w: take(out * inn),
        };
    }
    let (u, vt, s_post, s_pre) = (take(out * rank), take(inn * rank), take(out), take(inn));
    Param::Factor(HostFactor {
        out,
        inn,
        rank,
        u,
        vt,
        s_post,
        s_pre,
    })
}

fn save_state(dir: &str, pos: usize, ph: usize, fd: [bool; 3], bd: [bool; 4]) {
    let mut s = format!("{pos} {ph}");
    for v in fd.iter().chain(bd.iter()) {
        s.push_str(if *v { " 1" } else { " 0" });
    }
    s.push('\n');
    std::fs::write(format!("{dir}/state"), s).expect("state");
}

/// `(slot position, phase, front factored, block factored)`, the phase 0 = nonfact, 1 = ADMM,
/// 2 = fact.
fn load_state(dir: &str) -> Option<(usize, usize, [bool; 3], [bool; 4])> {
    let s = std::fs::read_to_string(format!("{dir}/state")).ok()?;
    let v: Vec<usize> = s
        .split_whitespace()
        .map(|t| t.parse().expect("state field"))
        .collect();
    assert_eq!(v.len(), 9, "state fields");
    let mut fd = [false; 3];
    let mut bd = [false; 4];
    for k in 0..3 {
        fd[k] = v[2 + k] == 1;
    }
    for k in 0..4 {
        bd[k] = v[5 + k] == 1;
    }
    Some((v[0], v[1], fd, bd))
}

fn save_params(dir: &str, p: &Params) {
    std::fs::create_dir_all(dir).expect("params dir");
    for k in 0..3 {
        let path = format!("{dir}/{}.param", FRONT_KEYS[k]);
        let v = if p.fd[k] {
            Param::Factor(p.ff[k].clone())
        } else {
            Param::Dense {
                out: p.fdims[k].0,
                inn: p.fdims[k].1,
                w: p.fw[k].clone(),
            }
        };
        save_param(&path, &v);
    }
    for k in 0..4 {
        let path = format!("{dir}/{}.param", BLOCK_KEYS[k]);
        let v = if p.bd[k] {
            Param::Factor(p.bf[k].clone())
        } else {
            Param::Dense {
                out: p.bdims[k].0,
                inn: p.bdims[k].1,
                w: p.bw[k].clone(),
            }
        };
        save_param(&path, &v);
    }
}

fn load_params(dir: &str) -> Params {
    let mut p = Params {
        fd: [false; 3],
        bd: [false; 4],
        fdims: [(0, 0); 3],
        bdims: [(0, 0); 4],
        fw: [Vec::new(), Vec::new(), Vec::new()],
        bw: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        ff: [(); 3].map(|()| ones_factor(0, 0, 0)),
        bf: [(); 4].map(|()| ones_factor(0, 0, 0)),
    };
    for k in 0..3 {
        match load_param(&format!("{dir}/{}.param", FRONT_KEYS[k])) {
            Param::Dense { out, inn, w } => {
                p.fdims[k] = (out, inn);
                p.fw[k] = w;
            }
            Param::Factor(h) => {
                p.fd[k] = true;
                p.fdims[k] = (h.out, h.inn);
                p.ff[k] = h;
            }
        }
    }
    for k in 0..4 {
        match load_param(&format!("{dir}/{}.param", BLOCK_KEYS[k])) {
            Param::Dense { out, inn, w } => {
                p.bdims[k] = (out, inn);
                p.bw[k] = w;
            }
            Param::Factor(h) => {
                p.bd[k] = true;
                p.bdims[k] = (h.out, h.inn);
                p.bf[k] = h;
            }
        }
    }
    p
}

/// The two rates the reference anneals (`compress_block.py:73-74`, `:182-185`), as multipliers on
/// [`LR_BASE`]: `c_t = lr_t / LR_BASE`, the same scalar for every group because each stage's
/// `eta_min` is `1e-4` times its group rate (`block.rs:835-843`).
fn sched_scaled(base: f64, eta_min: f64, t_max: usize) -> Vec<f64> {
    cosine_lr(base, eta_min, t_max, t_max)
        .into_iter()
        .map(|v| v / LR_BASE)
        .collect()
}

fn q32_f32(v: i64) -> f32 {
    (v as f64 / 4_294_967_296.0) as f32
}

/// End the process on the success path instead of racing the driver's exit.
///
/// This file allocates four tuner tapes and a layer tape (several hundred pipelines). After the test
/// body returns, the driver's compiler threads fault in `libnvidia-gpucomp` during process teardown
/// (`SIGSEGV` 4 of 4 runs on the bf16 stream path, `SIGSEGV`/hang 2 of 2 with the streams unrounded,
/// 0 of 2 with no tape allocated); the numbers and every assertion have already run and been
/// flushed line by line, so the exit is taken here rather than left to the driver.
fn finish() -> ! {
    std::io::Write::flush(&mut std::io::stdout()).expect("flush");
    unsafe { libc::_exit(0) }
}

fn f32_q32(v: f32) -> i64 {
    (f64::from(v) * 4_294_967_296.0).round() as i64
}

/// One round's knobs.
struct Opt {
    model: String,
    dump: String,
    tag: String,
    lo: usize,
    hi: usize,
    slots: Vec<usize>,
    bf16: bool,
    inject: bool,
    floor: f64,
    phases: usize,
}

fn opt() -> Opt {
    let range = env("APEIRON_NQ_BLOCKS", "0..1");
    let (a, b) = range.split_once("..").expect("a..b");
    let slots: Vec<usize> = env("APEIRON_NQ_SLOTS", "q,v,o,k,gate,up,down")
        .split(',')
        .map(|n| {
            let n = n.trim();
            let want = if matches!(n, "gate" | "up" | "down") {
                format!("mlp.{n}_proj")
            } else {
                format!("self_attn.{n}_proj")
            };
            SLOT_KEYS
                .iter()
                .position(|k| *k == want)
                .unwrap_or_else(|| panic!("APEIRON_NQ_SLOTS: {n:?}"))
        })
        .collect();
    Opt {
        model: env("APEIRON_NQ_MODEL", MODEL),
        dump: env("APEIRON_NQ_DUMP", DUMP),
        tag: env("APEIRON_NQ_TAG", "loop"),
        lo: a.parse().expect("a"),
        hi: b.parse().expect("b"),
        slots,
        bf16: env("APEIRON_NQ_STREAM", "bf16") != "f32",
        inject: env("APEIRON_NQ_ADMM", "b") == "a",
        floor: envf("APEIRON_NQ_FLOOR", 0.0),
        phases: envu("APEIRON_NQ_PHASES", usize::MAX),
    }
}

/// The reference's sequential block loop over blocks `lo..hi`, resumable per phase.
fn drive(engine: &Engine, cfg: &Config, o: &Opt) {
    let t_all = Instant::now();
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let (g, _mem) = GpuCtx::new(&vk).expect("gpu ctx");
    let (d, f) = (cfg.n_embd, cfg.n_ff);
    let ids = read_i32(&dump_at("calib_ids.i32"));
    let n_samples = envu("APEIRON_NQ_SAMPLES", 128);
    let seq = ids.len() / n_samples;
    let t = PER_STEP * seq;
    let chunks = t / N_CTX;
    let numel = (n_samples * seq * d) as f64;
    let rk = ranks(&o.dump);
    let pm = perms(&o.dump);
    let lag = t_all.elapsed().as_secs_f64();
    println!(
        "BT {:?} dump {n_samples} samples x {seq} tokens, d {d} f {f}, T {t}, {chunks} chunks, \
         {} draws loaded {lag:.1} s",
        o.tag,
        pm.len()
    );
    let dir = format!("{LOOP}/{}", o.tag);
    std::fs::create_dir_all(&dir).expect("loop dir");
    let cb = unsafe { g.vk.reusable_command_buffer() };
    let stage = unsafe {
        stage_new(
            &g,
            1 << 22,
            BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST,
        )
    };
    let mut xt = unsafe { Stream::new(&g, seq, d, n_samples) };
    let mut xs = unsafe { Stream::new(&g, seq, d, n_samples) };
    let mut yt = unsafe { Stream::new(&g, seq, d, n_samples) };
    let mut next = unsafe { Stream::new(&g, seq, d, n_samples) };
    // Block `lo`'s input streams: `s-{lo}` is block `lo − 1`'s advanced output (`s-0` the embedding).
    let s0_xt = format!("{dir}/s-{}.xt.f32", o.lo);
    let s0_xs = format!("{dir}/s-{}.xs.f32", o.lo);
    if Path::new(&s0_xt).exists() && Path::new(&s0_xs).exists() {
        unsafe {
            load_stream(&g, cb, &stage, &xt, &s0_xt);
            load_stream(&g, cb, &stage, &xs, &s0_xs);
        }
        println!("BT streams restored from {s0_xt}");
    } else {
        assert_eq!(o.lo, 0, "block {}: no saved input stream {s0_xt}", o.lo);
        let mut x0: Vec<f32> = Vec::with_capacity(n_samples * seq * d);
        for s in 0..n_samples {
            let toks: Vec<usize> = ids[s * seq..(s + 1) * seq]
                .iter()
                .map(|&x| x as usize)
                .collect();
            let rows = load_rows(engine, "token_embd.weight", cfg.vocab, d, &toks)
                .expect("embedding rows");
            x0.extend(rows.iter().map(|&v| round_bf16(q32_f32(v))));
        }
        unsafe { stream_upload(&g, xt.buf, &x0) };
        unsafe {
            submit_copies(&g, cb, &[(xt.buf, 0, xs.buf, 0, xt.words())]);
        }
        // The dump's own block-0 input, sample 0: the embedding path's check (`b0/x0.bf16`).
        let want = read_bf16(&dump_at(&format!("{B0}/x0.bf16")));
        let got = &x0[..seq * d];
        let (mut num, mut den) = (0f64, 0f64);
        for (i, &w) in want.iter().enumerate() {
            let gv = f64::from(got[i]);
            num += (gv - f64::from(w)).powi(2);
            den += f64::from(w).powi(2);
        }
        println!(
            "BT embed vs b0/x0.bf16 rel {:.3e} ({} elements)",
            (num / den).sqrt(),
            seq * d
        );
        if o.bf16 {
            unsafe {
                let all: Vec<usize> = (0..n_samples).collect();
                for c in all {
                    let mut at = 0;
                    let n = xt.chunk_words();
                    while at < n {
                        let take = stage.n.min(n - at);
                        let mut v = get_at(&g, cb, &stage, xs.buf, c * n + at, take);
                        for x in &mut v {
                            *x = round_bf16(*x);
                        }
                        put_at(&g, cb, &stage, xs.buf, c * n + at, &v);
                        at += take;
                    }
                }
            }
            println!("BT streams rounded to bf16");
        }
    }
    let mut fwd: Option<LayerFwd> = None;
    let mut big: Option<(BlockTune, FrontTune)> = None;
    let mut small: Option<(BlockTune, FrontTune)> = None;
    let mut round_t = Instant::now();
    for l in o.lo..o.hi {
        let host = LayerHost::from_engine(engine, cfg, l).expect("layer host");
        let (nh, nkv, hd, rot, theta) = match &host.mixer {
            Mixer::Softmax {
                nh,
                nkv,
                hd,
                rot,
                theta,
                ..
            } => (*nh, *nkv, *hd, *rot, *theta),
            Mixer::Affine { .. } => panic!("block {l}: not a softmax layer"),
        };
        let fs = FrontShape {
            chunks,
            n_ctx: N_CTX,
            d,
            nh,
            nkv,
            hd,
            rot,
            theta,
            eps: host.eps,
            ranks: [rk[l][0], rk[l][3], rk[l][1]],
        };
        let bs = BlockShape {
            t,
            d,
            a: host.a,
            f,
            ranks: [rk[l][2], rk[l][4], rk[l][5], rk[l][6]],
        };
        let fss = FrontShape {
            chunks: seq / N_CTX,
            ..fs
        };
        let bss = BlockShape { t: seq, ..bs };
        if fwd.is_none() {
            fwd = Some(unsafe { LayerFwd::new(&g, &host, N_CTX, chunks) });
        }
        let fw = fwd.as_ref().expect("fwd");
        unsafe { fw.load(&g, &host) };
        if big.is_none() {
            let b0 = unsafe { BlockTune::new(&g, bs, LR_BASE, LR_BASE, LR_BASE) };
            let f0 = unsafe { FrontTune::new(&g, fs, LR_BASE, LR_BASE) };
            let b1 = unsafe { BlockTune::new(&g, bss, LR_BASE, LR_BASE, LR_BASE) };
            let f1 = unsafe { FrontTune::new(&g, fss, LR_BASE, LR_BASE) };
            let (mut b0, mut f0, mut b1, mut f1) = (b0, f0, b1, f1);
            unsafe {
                b0.set_hyper(&g, AdamHyper::OPTIMI);
                f0.set_hyper(&g, AdamHyper::OPTIMI);
                b1.set_hyper(&g, AdamHyper::OPTIMI);
                f1.set_hyper(&g, AdamHyper::OPTIMI);
            }
            big = Some((b0, f0));
            small = Some((b1, f1));
        }
        let (bt, ft) = big.as_mut().expect("big tuners");
        let (bsm, fsm) = small.as_mut().expect("small tuners");
        let imp = stat(&o.dump, l, "mlp.down_proj", "o_norm");
        assert_eq!(imp.len(), d, "block {l}: importance");
        unsafe {
            bt.load_importance(&g, &imp);
            bsm.load_importance(&g, &imp);
        }
        let bund = Bund {
            g: &g,
            cb,
            st: &stage,
            xs: &xs,
            yt: &yt,
            next: &next,
            t,
            a: host.a,
            bf16: o.bf16,
        };
        let bundsm = Bund {
            g: &g,
            cb,
            st: &stage,
            xs: &xs,
            yt: &yt,
            next: &next,
            t: seq,
            a: host.a,
            bf16: o.bf16,
        };
        // `yt = fp_block(xt)`, one sample at a time (`compress_model.py:58-63`), bf16 on store.
        let t_yt = Instant::now();
        let (fx, _fao, fy) = fw.handles();
        for s0 in (0..n_samples).step_by(PER_STEP) {
            let ids4: Vec<usize> = (s0..s0 + PER_STEP).collect();
            unsafe {
                fw.run_dev(
                    &g,
                    &[chunk_into(&xt, &ids4, fx)],
                    &[chunk_from(&yt, &ids4, fy)],
                );
            }
            if o.bf16 {
                unsafe { round_ids(&g, cb, &stage, &yt, &ids4) };
            }
        }
        let yt_s = t_yt.elapsed().as_secs_f64();
        if l == 0 {
            let want = read_bf16(&dump_at(&format!("{B0}/yt0.bf16")));
            let mut got = vec![0f32; seq * d];
            let mut at = 0;
            while at < seq * d {
                let take = stage.n.min(seq * d - at);
                let v = unsafe { get_at(&g, cb, &stage, yt.buf, at, take) };
                got[at..at + take].copy_from_slice(&v);
                at += take;
            }
            let (mut num, mut den) = (0f64, 0f64);
            for (i, &w) in want.iter().enumerate() {
                num += (f64::from(got[i]) - f64::from(w)).powi(2);
                den += f64::from(w).powi(2);
            }
            println!("BT teacher vs b0/yt0.bf16 rel {:.3e}", (num / den).sqrt());
        }
        println!("BT l{l:02} teacher {yt_s:.1} s");
        let (g1, g2) = (host.attn_norm.clone(), host.ffn_norm.clone());
        let gn = match &host.mixer {
            Mixer::Softmax { gains, .. } => gains.clone(),
            Mixer::Affine { .. } => unreachable!(),
        };
        let dir_l = format!("{dir}/l{l}");
        std::fs::create_dir_all(&dir_l).expect("block dir");
        let resume = load_state(&dir_l);
        let (mut cur, mut cph, mut p) = if let Some((si, ph, fd, bd)) = resume {
            let mut p = load_params(&dir_l);
            p.fd = fd;
            p.bd = bd;
            println!("BT l{l:02} resume at slot {si} phase {ph}");
            (si, ph, p)
        } else {
            let p = teacher_params(&host, fs, bs);
            save_state(&dir_l, 0, 0, p.fd, p.bd);
            (0, 0, p)
        };
        let mut units = 0usize;
        let mut complete = true;
        'units: for si in 0..7 {
            if !o.slots.contains(&si) {
                continue;
            }
            for phase in 0..3 {
                if (si, phase) < (cur, cph) {
                    continue;
                }
                if units >= o.phases {
                    complete = false;
                    break 'units;
                }
                let t_u = Instant::now();
                match phase {
                    0 => {
                        let sched = sched_scaled(1e-4, 1e-8, 256);
                        let tf = [!p.fd[0], !p.fd[1], !p.fd[2]];
                        let tb = [!p.bd[0], !p.bd[1], !p.bd[2], !p.bd[3]];
                        unsafe {
                            engage(
                                &g,
                                bt,
                                ft,
                                &p,
                                (&g1, &gn, &g2),
                                &sched,
                                2.0 / PER_STEP as f32,
                                (tf, tb),
                            );
                        }
                        let mut sums = Vec::new();
                        for e in 0..8 {
                            let perm = &pm[2 * si * 8 + e].1;
                            let mut sum = 0.0;
                            for b0 in 0..n_samples / PER_STEP {
                                let ids4 = perm[b0 * PER_STEP..(b0 + 1) * PER_STEP].to_vec();
                                sum += unsafe { step(&bund, bt, ft, &ids4, tf.iter().any(|x| *x)) };
                            }
                            sums.push(sum);
                            println!(
                                "BT l{l:02} {} nonfact (Epoch {:02}/08) Block Loss: {:.4e}",
                                SLOT_KEYS[si],
                                e + 1,
                                sum / numel
                            );
                        }
                        unsafe { read_params(&g, &mut p, bt, ft) };
                        save_params(&dir_l, &p);
                        save_state(&dir_l, si, 1, p.fd, p.bd);
                        println!(
                            "BT l{l:02} {} nonfact {} steps {:.1} s",
                            SLOT_KEYS[si],
                            n_samples / PER_STEP * 8,
                            t_u.elapsed().as_secs_f64()
                        );
                    }
                    1 => {
                        let key = SLOT_KEYS[si];
                        let (out, inn, rank, w) = match (SLOT_FRONT[si], SLOT_BLOCK[si]) {
                            (Some(k), _) => {
                                let (oo, ii) = fs.dims(k);
                                (oo, ii, fs.ranks[k], unsafe { ft.weight(&g, k) })
                            }
                            (_, Some(k)) => {
                                let (oo, ii) = bs.dims(k);
                                (oo, ii, bs.ranks[k], unsafe { bt.weight(&g, k) })
                            }
                            _ => unreachable!(),
                        };
                        let i_norm: Vec<f64> = stat(&o.dump, l, key, "i_norm")
                            .iter()
                            .map(|&v| f64::from(v))
                            .collect();
                        let o_norm: Vec<f64> = stat(&o.dump, l, key, "o_norm")
                            .iter()
                            .map(|&v| f64::from(v))
                            .collect();
                        assert_eq!((i_norm.len(), o_norm.len()), (inn, out), "{key}: stats");
                        let t_a = Instant::now();
                        let hf = if o.inject {
                            assert_eq!(l, 0, "APEIRON_NQ_ADMM=a: the dump holds block 0");
                            dump_factor(&o.dump, key)
                        } else {
                            let wa: Vec<i64> = w.iter().map(|&v| f32_q32(v)).collect();
                            let nf = factorize_floor(
                                &wa,
                                out,
                                inn,
                                &i_norm,
                                &o_norm,
                                &NqConfig::reference(rank),
                                o.floor,
                            )
                            .expect("factorize");
                            let wh = nf.weight().expect("weight");
                            let mut raw = 0f64;
                            let mut orig = 0f64;
                            for (a2, b2) in wa.iter().zip(&wh) {
                                raw += ((q32_f32(*a2) - q32_f32(*b2)) as f64).powi(2);
                                orig += (q32_f32(*a2) as f64).powi(2);
                            }
                            let norm = raw / orig;
                            let s = t_a.elapsed().as_secs_f64();
                            println!(
                                "BT l{l:02} {key} ADMM weight recon error: raw={raw:.4}, \
                                 norm={norm:.4}, per_el={:.4e}, ADMM time={s:.2}s",
                                raw / (out * inn) as f64
                            );
                            host_from_nq(&nf).0
                        };
                        // The injected factor's own reconstruction: the convention check.
                        if o.inject {
                            let wa: Vec<i64> = w.iter().map(|&v| f32_q32(v)).collect();
                            let nf = nq_from_host(&hf, 0.0);
                            let wh = nf.weight().expect("weight");
                            let mut raw = 0f64;
                            let mut orig = 0f64;
                            for (a2, b2) in wa.iter().zip(&wh) {
                                raw += ((q32_f32(*a2) - q32_f32(*b2)) as f64).powi(2);
                                orig += (q32_f32(*a2) as f64).powi(2);
                            }
                            println!(
                                "BT l{l:02} {key} injected recon norm={:.4} raw={raw:.4} \
                                 (run4 q 0.2589 / v 0.3043), ADMM time=0.00s",
                                raw / orig
                            );
                        }
                        match (SLOT_FRONT[si], SLOT_BLOCK[si]) {
                            (Some(k), _) => {
                                p.fd[k] = true;
                                p.ff[k] = hf.clone();
                                unsafe { ft.load_factor(&g, k, &hf) };
                            }
                            (_, Some(k)) => {
                                p.bd[k] = true;
                                p.bf[k] = hf.clone();
                                unsafe { bt.load_factor(&g, k, &hf) };
                            }
                            _ => unreachable!(),
                        }
                        unsafe {
                            bt.set_stage(&g, p.bd, [false; 4]);
                            ft.set_stage(&g, p.fd, [false; 3]);
                        }
                        save_params(&dir_l, &p);
                        save_state(&dir_l, si, 2, p.fd, p.bd);
                    }
                    _ => {
                        let sched = sched_scaled(1e-5, 1e-9, 1024);
                        let mut tf = [false; 3];
                        let mut tb = [false; 4];
                        if let Some(k) = SLOT_FRONT[si] {
                            tf[k] = true;
                        }
                        if let Some(k) = SLOT_BLOCK[si] {
                            tb[k] = true;
                        }
                        unsafe {
                            engage(&g, bsm, fsm, &p, (&g1, &gn, &g2), &sched, 2.0, (tf, tb));
                        }
                        for e in 0..8 {
                            let perm = &pm[(2 * si + 1) * 8 + e].1;
                            let mut sum = 0.0;
                            for i in 0..n_samples {
                                sum += unsafe {
                                    step(&bundsm, bsm, fsm, &[perm[i]], tf.iter().any(|x| *x))
                                };
                            }
                            println!(
                                "BT l{l:02} {} fact (Epoch {:02}/08) Block Loss: {:.4e}",
                                SLOT_KEYS[si],
                                e + 1,
                                sum / numel
                            );
                        }
                        // `finalize` (`modules/linear.py:264-283`): the latents harden to their
                        // sign, `sign(0) = +1`.
                        let hard = |h: &mut HostFactor| {
                            for v in h.u.iter_mut().chain(h.vt.iter_mut()) {
                                *v = if *v < 0.0 { -1.0 } else { 1.0 };
                            }
                        };
                        match (SLOT_FRONT[si], SLOT_BLOCK[si]) {
                            (Some(k), _) => {
                                let mut h = unsafe { fsm.factors(&g)[k].clone() };
                                hard(&mut h);
                                p.ff[k] = h.clone();
                                unsafe { fsm.load_factor(&g, k, &h) };
                            }
                            (_, Some(k)) => {
                                let mut h = unsafe { bsm.factors(&g)[k].clone() };
                                hard(&mut h);
                                p.bf[k] = h.clone();
                                unsafe { bsm.load_factor(&g, k, &h) };
                            }
                            _ => unreachable!(),
                        }
                        unsafe { read_params(&g, &mut p, bsm, fsm) };
                        save_params(&dir_l, &p);
                        save_state(&dir_l, si, 3, p.fd, p.bd);
                        println!(
                            "BT l{l:02} {} fact {} steps {:.1} s",
                            SLOT_KEYS[si],
                            n_samples * 8,
                            t_u.elapsed().as_secs_f64()
                        );
                    }
                }
                units += 1;
                cur = si;
                cph = phase + 1;
            }
        }
        let done = p.fd.iter().all(|x| *x) && p.bd.iter().all(|x| *x) && complete;
        if done {
            // The student stream advances through the finished block (`compress_model.py:103-110`).
            unsafe {
                engage(
                    &g,
                    bt,
                    ft,
                    &p,
                    (&g1, &gn, &g2),
                    &sched_scaled(1e-7, 1e-7, 1),
                    2.0 / PER_STEP as f32,
                    ([false; 3], [false; 4]),
                );
            }
            let t_adv = Instant::now();
            for s0 in (0..n_samples).step_by(PER_STEP) {
                let ids4: Vec<usize> = (s0..s0 + PER_STEP).collect();
                unsafe { eval_pass(&bund, bt, ft, &ids4) };
            }
            std::mem::swap(&mut xt, &mut yt);
            std::mem::swap(&mut xs, &mut next);
            let adv_s = t_adv.elapsed().as_secs_f64();
            unsafe {
                save_stream(&g, cb, &stage, &xt, &format!("{dir}/s-{}.xt.f32", l + 1));
                save_stream(&g, cb, &stage, &xs, &format!("{dir}/s-{}.xs.f32", l + 1));
            }
            std::fs::write(format!("{dir_l}/done"), b"1\n").expect("done");
            println!("BT l{l:02} advanced {adv_s:.1} s");
        } else {
            println!(
                "BT l{l:02} not advanced: slots {:?} of the reference's seven are not all factors",
                o.slots
            );
        }
        println!(
            "BT l{l:02} round total {:.1} s",
            round_t.elapsed().as_secs_f64()
        );
        round_t = Instant::now();
        // Block `l + 1` reads block `l`'s advanced streams; an unfinished block ends the round.
        if !done {
            break;
        }
    }
    finish();
}

/// The host-only arm (b): our ADMM on the dump's `b0/<key>.w_in.f32` and the dump's statistics,
/// beside the reference's own `norm` for the same input.
fn arm_b0(o: &Opt) {
    let rk = ranks(&o.dump);
    for si in [0usize, 1] {
        let key = SLOT_KEYS[si];
        let w = read_f32(&dump_at(&format!("{B0}/{key}.w_in.f32")));
        let f64s = |which: &str| -> Vec<f64> {
            stat(&o.dump, 0, key, which)
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        };
        let (i_norm, o_norm) = (f64s("i_norm"), f64s("o_norm"));
        let (inn, out) = (i_norm.len(), o_norm.len());
        assert_eq!(w.len(), inn * out, "{key}: w_in");
        let (rank, d0) = (rk[0][si], o.floor);
        let wa: Vec<i64> = w.iter().map(|&v| f32_q32(v)).collect();
        let orig: f64 = wa.iter().map(|&a| (q32_f32(a) as f64).powi(2)).sum();
        let t = Instant::now();
        let nf = factorize_floor(
            &wa,
            out,
            inn,
            &i_norm,
            &o_norm,
            &NqConfig::reference(rank),
            d0,
        )
        .expect("factorize");
        let s = t.elapsed().as_secs_f64();
        let ours = recon(wa.iter().zip(&nf.weight().expect("weight")));
        let theirs = dump_factor(&o.dump, key);
        let theirs = recon(
            wa.iter()
                .zip(&nq_from_host(&theirs, 0.0).weight().expect("weight")),
        );
        println!(
            "BT arm-b {key} rank {rank} ours norm={:.4} theirs(binary) norm={:.4} \
             (run4 float 0.2589/0.3043) ratio {:.3}, admm_loss {:.5e}, {s:.2} s",
            ours / orig,
            theirs / orig,
            ours / theirs,
            nf.admm_loss
        );
    }
}

/// `Σ(W − Ŵ)²` over a Q32.32 word pair, in f64.
fn recon<'a>(pairs: impl Iterator<Item = (&'a i64, &'a i64)>) -> f64 {
    pairs
        .map(|(a, b)| ((q32_f32(*a) - q32_f32(*b)) as f64).powi(2))
        .sum()
}

/// `NanoQuant`'s sequential block loop on the device: `APEIRON_NQ_BLOCKS` (`a..b`, default `0..1`),
/// `APEIRON_NQ_SLOTS` (`q,v` for the gate) and `APEIRON_NQ_PHASES` per round; `APEIRON_NQ_ARM=b0`
/// runs the host-only ADMM arm.
#[test]
#[ignore = "measurement: the reference's block loop whole; opens the teacher model"]
fn block_loop() {
    let o = opt();
    if env("APEIRON_NQ_ARM", "") == "b0" {
        arm_b0(&o);
        finish();
    }
    let engine = Engine::open(&o.model).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    drive(&engine, &cfg, &o);
}
