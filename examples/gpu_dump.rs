//! GPU forward-pass dump: the output of `attn_cut_probe`'s `run_dump` (`KV_DUMP=<dir> KV_T=<tokens>`),
//! computed in f32 on the Vulkan device, for Qwen2-family GGUF models (head_dim 64 or 128).
//! Kernels: `gpu_dump_spirv/`.
//!
//! Run: `KV_DUMP=<dir> KV_T=<tokens> gpu_dump [model.gguf]`
//!
//! Files in `<dir>`: `ids.bin` (u32 LE), `cs.bin` (`KV_T x head_dim/2` (cos, sin) f64 LE), and per layer
//! `q{l}.bin` (`KV_T x n_head x hd`), `k{l}.bin`, `v{l}.bin` (`KV_T x n_head_kv x hd`), f32 LE and
//! unrotated. `KV_QTAIL=t1,t2,..` (multiples of 64) writes in `q{l}.bin` only rows `[t-64, t)` for each
//! listed `t`, concatenated; k and v stay full.
//!
//! Per layer and token chunk of `KV_CHUNK` rows (default 8192, multiple of 64): rmsnorm, q/k/v GEMMs with
//! bias, RoPE (half-split, f64-derived table rounded to f32), causal flash attention (f32, online
//! softmax) against the device-resident rotated K and V of rows `[0, chunk end)`, `Wo`, rmsnorm, SwiGLU
//! FFN. The residual streams through `x.bin` (layer input) and `x.next.bin` (layer output); `next.txt`
//! is `l` (layer `l` from its start) or `l c` (layer `l`, next chunk `c`). An invocation stops after the
//! chunk that crosses `KV_BUDGET` seconds (default 420) and exits 3; rerun to continue. Output rows are
//! written at their file offset, so a resumed run is byte-identical to an uninterrupted one.
//!
//! `GPU_DUMP_CMP=<refdir>`: after each layer, compare the first `min(rows)` rows of q, k, v with the
//! reference dump (relative L2, bar `1e-4`); without `KV_T` it compares the finished layers of `KV_DUMP`
//! only. Exit 1 on a layer over the bar.
//!
//! Cost per layer: GEMM `2 KV_T (d (n_head hd + 2 n_kv hd) + n_head hd d + 3 d n_ff)` flops; attention
//! `2 * 2 n_head hd KV_T (KV_T + 1) / 2` flops. Device memory: one layer of weights, rotated K and V
//! `2 KV_T n_kv hd` f32, the rope table `KV_T hd` f32, and chunk-sized activations including the
//! `KV_CHUNK x n_ff` gate and up planes. Host memory: the rope table and the staging buffers.

#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::undocumented_unsafe_blocks,
    clippy::missing_safety_doc,
    reason = "measurement harness: numeric casts, indexed loops and the Vulkan FFI wrapper are required"
)]

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    app::main()
}

#[cfg(feature = "gpu")]
mod app {
    use std::error::Error;
    use std::os::unix::fs::FileExt;
    use std::path::{Path, PathBuf};
    use std::process::ExitCode;
    use std::time::Instant;

    use apeiron::engine::dequant_row_fixed;
    use apeiron::engine::loader::Engine;
    use apeiron::format::bpe::Bpe;
    use apeiron::gpu::vulkan::*;

    const MODEL: &str = concat!(env!("HOME"), "/models/gguf/qwen05b-Q4_K_M.gguf");
    const TOKENIZER: &str = concat!(env!("HOME"), "/models/qwen2.5-0.5b/tokenizer.json");
    const PROSE: &str = concat!(env!("HOME"), "/Workspace/fold/hashtrinity/prose8.txt");
    const ONE: f64 = 4_294_967_296.0;
    const NONE: u32 = 0xFFFF_FFFF;
    const BAR: f64 = 1e-4;

    static GEMM: &[u8] = include_bytes!("gpu_dump_spirv/dump_gemm.spv");
    static RMS: &[u8] = include_bytes!("gpu_dump_spirv/dump_rms.spv");
    static ROPE: &[u8] = include_bytes!("gpu_dump_spirv/dump_rope.spv");
    static SILU: &[u8] = include_bytes!("gpu_dump_spirv/dump_silu.spv");
    static ATTN64: &[u8] = include_bytes!("gpu_dump_spirv/dump_attn64.spv");
    static ATTN128: &[u8] = include_bytes!("gpu_dump_spirv/dump_attn128.spv");

    type R<T> = Result<T, Box<dyn Error>>;

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    fn q32_vec(v: &[i64]) -> Vec<f32> {
        v.iter().map(|&x| (x as f64 / ONE) as f32).collect()
    }

    /// Dequantised `name` as f32, `n_out x n_in` row-major, and its `(n_in, n_out)`.
    /// Time O(n_in n_out) element decodes; space 12 bytes per element transiently (i64 plus f32).
    fn weight(e: &Engine, name: &str) -> R<(Vec<f32>, usize, usize)> {
        let t = e
            .table()
            .get(name)
            .ok_or_else(|| format!("missing {name}"))?;
        let n_in = t.dims[0] as usize;
        let n_out = t.dims[1] as usize;
        let nb = t.n_bytes.ok_or("no byte count")?;
        let raw = e.payload(t.byte_range.start, nb)?;
        let mut out = Vec::with_capacity(n_in * n_out);
        dequant_row_fixed(t.ggml_type, raw, n_in * n_out, &mut out)?;
        if out.len() != n_in * n_out {
            return Err(format!("{name}: dequantised {} != {}", out.len(), n_in * n_out).into());
        }
        Ok((q32_vec(&out), n_in, n_out))
    }

    /// One shader invocation shape: pipeline, bound descriptor set, and the host-mapped parameter words.
    struct Role {
        layout: Handle,
        set: Handle,
        pipe: Handle,
        params: *mut u32,
    }

    struct Ctx {
        vk: Vk,
        cb: CommandBuffer,
        vram: u64,
        host: u64,
    }

    const HV: u32 = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
    const USAGE: u32 = BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST;

    impl Ctx {
        unsafe fn dev(&mut self, bytes: u64) -> Handle {
            let b = bytes.max(256);
            let (buf, _) = self
                .vk
                .buffer(b, USAGE, MEMORY_DEVICE_LOCAL)
                .expect("device-local memory type");
            self.vram += b;
            buf
        }

        unsafe fn host(&mut self, bytes: u64, cached: bool) -> (Handle, *mut u8) {
            let b = bytes.max(256);
            let r = if cached {
                self.vk.buffer(b, USAGE, HV | MEMORY_HOST_CACHED)
            } else {
                None
            };
            let (buf, mem) = r
                .or_else(|| self.vk.buffer(b, USAGE, HV))
                .expect("host-visible memory type");
            self.host += b;
            (buf, self.vk.map(mem).cast::<u8>())
        }

        /// Record `f` into the reusable command buffer, submit, wait; returns seconds.
        unsafe fn run(&self, f: impl FnOnce(&Vk, CommandBuffer)) -> f64 {
            let vk = &self.vk;
            let t = Instant::now();
            vk.reset_cb(self.cb);
            vk.begin(self.cb);
            f(vk, self.cb);
            check((vk.end_cb)(self.cb), "vkEndCommandBuffer");
            vk.submit_wait(self.cb);
            t.elapsed().as_secs_f64()
        }

        /// Role over `bufs` (binding 0 is the parameter block, allocated here); the pipeline for a kind
        /// is created on the first role and reused.
        unsafe fn role(&mut self, pipe: &mut Option<Handle>, code: &[u8], bufs: &[Handle]) -> Role {
            let (pb, p) = self.host(256, false);
            let mut all = vec![pb];
            all.extend_from_slice(bufs);
            let (layout, set) = self.vk.ssbo_set(&all);
            let pipe = *pipe.get_or_insert_with(|| self.vk.pipeline(code, layout));
            Role {
                layout,
                set,
                pipe,
                params: p.cast::<u32>(),
            }
        }
    }

    unsafe fn setp(r: &Role, v: &[u32]) {
        for (i, x) in v.iter().enumerate() {
            r.params.add(i).write_volatile(*x);
        }
    }

    unsafe fn disp(vk: &Vk, cb: CommandBuffer, r: &Role, g: [u32; 3]) {
        (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, r.pipe);
        (vk.bind_sets)(
            cb,
            PIPELINE_BIND_POINT_COMPUTE,
            r.layout,
            0,
            1,
            &r.set,
            0,
            std::ptr::null(),
        );
        (vk.dispatch)(cb, g[0], g[1], g[2]);
    }

    unsafe fn bar(vk: &Vk, cb: CommandBuffer) {
        vk.barrier(
            cb,
            STAGE_COMPUTE | STAGE_TRANSFER,
            STAGE_COMPUTE | STAGE_TRANSFER,
            ACCESS_SHADER_WRITE | ACCESS_TRANSFER_WRITE,
            ACCESS_SHADER_READ | ACCESS_SHADER_WRITE | ACCESS_TRANSFER_READ | ACCESS_TRANSFER_WRITE,
        );
    }

    unsafe fn copy(
        vk: &Vk,
        cb: CommandBuffer,
        src: Handle,
        dst: Handle,
        so: u64,
        d_o: u64,
        n: u64,
    ) {
        let r = BufferCopy {
            src_offset: so,
            dst_offset: d_o,
            size: n,
        };
        (vk.copy_buffer)(cb, src, dst, 1, &r);
    }

    fn host_barrier(vk: &Vk, cb: CommandBuffer) {
        unsafe {
            vk.barrier(
                cb,
                STAGE_COMPUTE | STAGE_TRANSFER,
                STAGE_HOST,
                ACCESS_SHADER_WRITE | ACCESS_TRANSFER_WRITE,
                ACCESS_HOST_READ,
            );
        }
    }

    /// Copy `data` into `dst` at byte offset `off` through the staging buffer `stg` (`stg_n` bytes).
    /// Time O(bytes) (one submit per `stg_n` bytes); space `stg_n`.
    unsafe fn upload(
        cx: &Ctx,
        stg: Handle,
        stg_p: *mut u8,
        stg_n: usize,
        data: &[u8],
        dst: Handle,
        off: u64,
    ) {
        let mut done = 0usize;
        while done < data.len() {
            let n = (data.len() - done).min(stg_n);
            std::ptr::copy_nonoverlapping(data.as_ptr().add(done), stg_p, n);
            cx.run(|vk, cb| copy(vk, cb, stg, dst, 0, off + done as u64, n as u64));
            done += n;
        }
    }

    /// Streamed relative L2 and max abs error of `a` against `b` over `n` f32 values from byte offsets
    /// `ao`, `bo`. Time O(n); space one 8 MiB block per file.
    fn rel_l2(fa: &std::fs::File, ao: u64, fb: &std::fs::File, bo: u64, n: usize) -> R<(f64, f64)> {
        const BLK: usize = 2 << 20;
        let (mut num, mut den, mut mx) = (0f64, 0f64, 0f64);
        let (mut ba, mut bb) = (vec![0u8; BLK * 4], vec![0u8; BLK * 4]);
        let mut done = 0usize;
        while done < n {
            let m = (n - done).min(BLK);
            fa.read_exact_at(&mut ba[..m * 4], ao + (done * 4) as u64)?;
            fb.read_exact_at(&mut bb[..m * 4], bo + (done * 4) as u64)?;
            for i in 0..m {
                let x = f64::from(f32::from_le_bytes([
                    ba[4 * i],
                    ba[4 * i + 1],
                    ba[4 * i + 2],
                    ba[4 * i + 3],
                ]));
                let y = f64::from(f32::from_le_bytes([
                    bb[4 * i],
                    bb[4 * i + 1],
                    bb[4 * i + 2],
                    bb[4 * i + 3],
                ]));
                let d = x - y;
                num += d * d;
                den += y * y;
                mx = mx.max(d.abs());
            }
            done += m;
        }
        Ok(((num / den.max(1e-300)).sqrt(), mx))
    }

    fn layer_of(dir: &Path) -> usize {
        std::fs::read_to_string(dir.join("next.txt"))
            .ok()
            .and_then(|s| s.split_whitespace().next().and_then(|t| t.parse().ok()))
            .unwrap_or(0)
    }

    /// Compare layer `l` of `dir` with `refd`; prints one line; returns the worst relative L2 of q, k, v.
    /// Time O(compared bytes); space two 8 MiB blocks.
    fn cmp_layer(
        dir: &Path,
        refd: &Path,
        l: usize,
        (nq, nk): (usize, usize),
        qtail: &[usize],
    ) -> R<f64> {
        let mut worst = 0f64;
        let mut line = format!("cmp layer {l}:");
        for (tag, w) in [("q", nq), ("k", nk), ("v", nk)] {
            let n = format!("{tag}{l}.bin");
            let (fo, fr) = (
                std::fs::File::open(dir.join(&n))?,
                std::fs::File::open(refd.join(&n))?,
            );
            let (ro, rr) = (
                fo.metadata()?.len() as usize / (w * 4),
                fr.metadata()?.len() as usize / (w * 4),
            );
            let (mut num_sq, mut den_sq, mut mx) = (0f64, 0f64, 0f64);
            // segments: (rows in ours at row0, rows in ref at row0, count)
            let segs: Vec<(usize, usize, usize)> = if tag == "q" && !qtail.is_empty() {
                qtail
                    .iter()
                    .enumerate()
                    .filter(|&(_, &t)| t <= rr)
                    .map(|(i, &t)| (i * 64, t - 64, 64))
                    .collect()
            } else {
                vec![(0, 0, ro.min(rr))]
            };
            let rows: usize = segs.iter().map(|s| s.2).sum();
            for (a, b, c) in segs {
                let (e, m) = rel_l2(&fo, (a * w * 4) as u64, &fr, (b * w * 4) as u64, c * w)?;
                // weight each segment by its reference energy: recompute den from e is not possible,
                // so accumulate squared relative errors (equal-size segments); max over segments is reported.
                num_sq += e * e;
                den_sq += 1.0;
                mx = mx.max(m);
            }
            let e = (num_sq / den_sq.max(1.0)).sqrt();
            worst = worst.max(e);
            line += &format!("  {tag} rel-L2 {e:.3e} (max abs {mx:.3e}, {rows} rows)");
        }
        println!("{line}{}", if worst > BAR { "  FAIL" } else { "" });
        Ok(worst)
    }

    fn parse_qtail(tn: usize) -> R<Vec<usize>> {
        let mut v: Vec<usize> = match std::env::var("KV_QTAIL") {
            Ok(s) => s
                .split(',')
                .map(|x| x.trim().parse::<usize>())
                .collect::<Result<_, _>>()?,
            Err(_) => Vec::new(),
        };
        v.sort_unstable();
        v.dedup();
        for &t in &v {
            if t < 64 || t % 64 != 0 || t > tn {
                return Err(format!(
                    "KV_QTAIL entry {t}: need a multiple of 64 in [64, KV_T={tn}]"
                )
                .into());
            }
        }
        Ok(v)
    }

    pub fn main() -> ExitCode {
        match run() {
            Ok(code) => code,
            Err(e) => {
                eprintln!("gpu_dump: {e}");
                ExitCode::FAILURE
            }
        }
    }

    fn rss_kb(key: &str) -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with(key))
                    .and_then(|l| l.split_whitespace().nth(1).and_then(|x| x.parse().ok()))
            })
            .unwrap_or(0)
    }

    fn vram_used_mib() -> Option<u64> {
        let o = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;
        String::from_utf8_lossy(&o.stdout)
            .trim()
            .lines()
            .next()?
            .trim()
            .parse()
            .ok()
    }

    /// Dump driver. Time per layer: weight decode O(layer params), GEMM `2 KV_T (layer params)` flops,
    /// attention `~2 n_head hd KV_T^2` flops. Space: see the module header.
    fn run() -> R<ExitCode> {
        let path = std::env::args().nth(1).unwrap_or_else(|| MODEL.to_string());
        let dir = PathBuf::from(std::env::var_os("KV_DUMP").ok_or("KV_DUMP not set")?);
        let refd = std::env::var_os("GPU_DUMP_CMP").map(PathBuf::from);
        let eng = Engine::open(&path)?;
        let cfg = &eng.config;
        let (nl, nh, nkv, d, nff, hd) = (
            cfg.n_layer,
            cfg.n_head,
            cfg.n_head_kv,
            cfg.n_embd,
            cfg.n_ff,
            cfg.head_dim,
        );
        let (nq, nk) = (nh * hd, nkv * hd);
        let half = hd / 2;
        let Ok(tn_s) = std::env::var("KV_T") else {
            // compare-only
            let refd = refd.ok_or("KV_T not set and GPU_DUMP_CMP not set")?;
            let qtail = parse_qtail(usize::MAX / 2)?;
            let ours = layer_of(&dir);
            let theirs = layer_of(&refd);
            let n = ours.min(theirs);
            println!(
                "compare-only: ours finished {ours} layers, reference {theirs}; comparing {n}"
            );
            let mut worst = 0f64;
            for l in 0..n {
                worst = worst.max(cmp_layer(&dir, &refd, l, (nq, nk), &qtail)?);
            }
            println!(
                "max rel-L2 over {n} layers: {worst:.3e} (bar {BAR:.0e}): {}",
                if worst <= BAR { "PASS" } else { "FAIL" }
            );
            return Ok(if worst <= BAR && n > 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            });
        };
        let tn: usize = tn_s.parse()?;
        assert!(
            hd == 64 || hd == 128,
            "head_dim {hd}: only 64 and 128 have an attention kernel"
        );
        assert!(tn.is_multiple_of(64), "KV_T must be a multiple of 64");
        for (n, v) in [("nq", nq), ("nk", nk), ("d", d), ("n_ff", nff)] {
            assert!(
                v % 128 == 0,
                "{n} = {v} must be a multiple of 128 (GEMM tile)"
            );
        }
        let budget = std::env::var("KV_BUDGET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(420.0f64);
        let t0 = Instant::now();
        let cn = env_usize("KV_CHUNK", 8192).min(tn);
        assert!(cn.is_multiple_of(64), "KV_CHUNK must be a multiple of 64");
        let nc = tn.div_ceil(cn);
        let rows = |c: usize| c * cn..((c + 1) * cn).min(tn);
        let qtail = parse_qtail(tn)?;
        std::fs::create_dir_all(&dir)?;

        // ---- tokens and rope table ----
        let bpe = Bpe::from_tokenizer_json(TOKENIZER)?;
        let raw = std::fs::read(PROSE)?;
        let text = String::from_utf8_lossy(&raw[..(tn * 8).min(raw.len())]).into_owned();
        let mut ids = bpe.encode(&text)?;
        if ids.len() < tn {
            return Err(format!("dump prompt: {} tokens < {tn}", ids.len()).into());
        }
        ids.truncate(tn);
        let idb: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
        let hex: String = apeiron::sha256::sha256(&idb)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        println!("dump: {tn} tokens, ids sha256 {hex}");
        let next_path = dir.join("next.txt");
        let (start_l, start_c) = match std::fs::read_to_string(&next_path) {
            Ok(s) => {
                let f: Vec<usize> = s
                    .split_whitespace()
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
                match f[..] {
                    [l] => (l, 0),
                    [l, c] => (l, c),
                    _ => return Err(format!("next.txt: {s:?}").into()),
                }
            }
            Err(_) => (0, 0),
        };
        let fresh = (start_l, start_c) == (0, 0);
        let rope = eng.rope_table().map_err(|e| format!("{e:?}"))?;
        let mut cs32 = vec![0f32; tn * half * 2];
        let mut cs64: Vec<u8> = Vec::with_capacity(if fresh { tn * half * 16 } else { 0 });
        for p in 0..tn {
            let mut x = vec![0i64; hd];
            for xj in x.iter_mut().take(half) {
                *xj = ONE as i64;
            }
            rope.rotate_one(&mut x, p as u32)
                .map_err(|e| format!("rope position {p}: {e:?}"))?;
            for j in 0..half {
                let (c, s) = (x[j] as f64 / ONE, x[j + half] as f64 / ONE);
                cs32[(p * half + j) * 2] = c as f32;
                cs32[(p * half + j) * 2 + 1] = s as f32;
                if fresh {
                    cs64.extend_from_slice(&c.to_le_bytes());
                    cs64.extend_from_slice(&s.to_le_bytes());
                }
            }
        }
        let xpath = dir.join("x.bin");
        if fresh {
            std::fs::write(dir.join("ids.bin"), &idb)?;
            std::fs::write(dir.join("cs.bin"), &cs64)?;
            drop(cs64);
            let xf = std::fs::File::create(&xpath)?;
            xf.set_len((tn * d * 4) as u64)?;
            for c in 0..nc {
                let r = rows(c);
                let mut b = Vec::with_capacity(r.len() * d * 4);
                for &id in &ids[r.clone()] {
                    for v in q32_vec(&eng.token_embedding(id)?) {
                        b.extend_from_slice(&v.to_le_bytes());
                    }
                }
                xf.write_all_at(&b, (r.start * d * 4) as u64)?;
            }
        }
        println!("resume at layer {start_l} chunk {start_c} of {nl} layers, {nc} chunks of {cn}; setup {:.1}s", t0.elapsed().as_secs_f64());

        // ---- device ----
        let vk = unsafe { Vk::load()? };
        println!("device: {}", vk.device_name);
        let cb = unsafe { vk.reusable_command_buffer() };
        let mut cx = Ctx {
            vk,
            cb,
            vram: 0,
            host: 0,
        };
        let f4 = 4u64;
        let (cnu, du, nqu, nku, nffu, tnu) = (
            cn as u64, d as u64, nq as u64, nk as u64, nff as u64, tn as u64,
        );
        let w_floats = (nq * d + 2 * nk * d + d * nq + 3 * d * nff) as u64;
        let sb_floats = ((2 * d + nq + 2 * nk) as u64).div_ceil(4) * 4;
        let stg_n = (128usize << 20).max(cn * d * 4).max(2 * cn * nk * 4);
        let rb_n = (cn * (2 * nk + nq).max(d) * 4) as u64;
        let (wbuf, sb, csb, xb, xn, sq, sk, sv, qr, kd, vd, at, gb, ub) = unsafe {
            (
                cx.dev(w_floats * f4),
                cx.dev(sb_floats * f4),
                cx.dev(tnu * half as u64 * 8),
                cx.dev(cnu * du * f4),
                cx.dev(cnu * du * f4),
                cx.dev(cnu * nqu * f4),
                cx.dev(cnu * nku * f4),
                cx.dev(cnu * nku * f4),
                cx.dev(cnu * nqu * f4),
                cx.dev(tnu * nku * f4),
                cx.dev(tnu * nku * f4),
                cx.dev(cnu * nqu * f4),
                cx.dev(cnu * nffu * f4),
                cx.dev(cnu * nffu * f4),
            )
        };
        let (stg, stg_p) = unsafe { cx.host(stg_n as u64, false) };
        let (rb, rb_p) = unsafe { cx.host(rb_n, true) };
        let (mut pg, mut pr, mut pp, mut ps, mut pa) = (None, None, None, None, None);
        let attn_code = if hd == 64 { ATTN64 } else { ATTN128 };
        let (rms_a, gq, gk, gv, ropeq, ropek, attn, go, rms_b, gg, gu, silu, gd) = unsafe {
            (
                cx.role(&mut pr, RMS, &[xb, sb, xn]),
                cx.role(&mut pg, GEMM, &[wbuf, sb, xn, sq]),
                cx.role(&mut pg, GEMM, &[wbuf, sb, xn, sk]),
                cx.role(&mut pg, GEMM, &[wbuf, sb, xn, sv]),
                cx.role(&mut pp, ROPE, &[csb, sq, qr]),
                cx.role(&mut pp, ROPE, &[csb, sk, kd]),
                cx.role(&mut pa, attn_code, &[qr, kd, vd, at]),
                cx.role(&mut pg, GEMM, &[wbuf, sb, at, xb]),
                cx.role(&mut pr, RMS, &[xb, sb, xn]),
                cx.role(&mut pg, GEMM, &[wbuf, sb, xn, gb]),
                cx.role(&mut pg, GEMM, &[wbuf, sb, xn, ub]),
                cx.role(&mut ps, SILU, &[gb, ub]),
                cx.role(&mut pg, GEMM, &[wbuf, sb, gb, xb]),
            )
        };
        unsafe {
            let cs8: &[u8] = std::slice::from_raw_parts(cs32.as_ptr().cast::<u8>(), cs32.len() * 4);
            upload(&cx, stg, stg_p, stg_n, cs8, csb, 0);
        }
        drop(cs32);
        let eps_bits = (cfg.rms_eps as f32).to_bits();
        let scale_bits = ((1.0 / (hd as f64).sqrt()) as f32).to_bits();
        let vecs = eng.decode_vectors();
        println!(
            "allocated: device {:.2} GiB, host-visible {:.2} MiB",
            cx.vram as f64 / 1073741824.0,
            cx.host as f64 / 1048576.0
        );

        let gemm_flops =
            |tok: usize, n_in: usize, n_out: usize| 2.0 * tok as f64 * n_in as f64 * n_out as f64;
        let attn_flops = 2.0 * 2.0 * nh as f64 * hd as f64 * (tn as f64) * (tn as f64 + 1.0) / 2.0;
        let layer_gemm =
            gemm_flops(tn, d, nq + 2 * nk) + gemm_flops(tn, nq, d) + 3.0 * gemm_flops(tn, d, nff);
        let mut worst = 0f64;
        let mut vram_peak = 0u64;
        let xnext = dir.join("x.next.bin");
        for l in start_l..nl {
            let tl = Instant::now();
            let xf = std::fs::File::open(&xpath)?;
            if xf.metadata()?.len() != (tn * d * 4) as u64 {
                return Err("x.bin size".into());
            }
            let c0 = if l == start_l { start_c } else { 0 };
            if c0 >= nc {
                return Err(format!("next.txt chunk {c0} >= {nc}").into());
            }
            // ---- layer weights ----
            let norm = |i: usize| -> R<Vec<f32>> {
                Ok(q32_vec(
                    vecs.norms_of(l)
                        .and_then(|n| n[i].as_deref())
                        .ok_or("norm")?,
                ))
            };
            let (g1, g2) = (norm(0)?, norm(1)?);
            let bias = |i: usize| vecs.biases_of(l).and_then(|b| b[i].as_deref()).map(q32_vec);
            let (bq, bkk, bv) = (bias(0), bias(1), bias(2));
            let mut sbh = vec![0f32; sb_floats as usize];
            sbh[..d].copy_from_slice(&g1);
            sbh[d..2 * d].copy_from_slice(&g2);
            let (o_bq, o_bk, o_bv) = (2 * d, 2 * d + nq, 2 * d + nq + nk);
            for (o, b, n) in [(o_bq, &bq, nq), (o_bk, &bkk, nk), (o_bv, &bv, nk)] {
                if let Some(b) = b {
                    assert_eq!(b.len(), n);
                    sbh[o..o + n].copy_from_slice(b);
                }
            }
            let bo = |o: usize, b: &Option<Vec<f32>>| if b.is_some() { o as u32 } else { NONE };
            let (wq_o, wk_o, wv_o) = (0usize, nq * d, nq * d + nk * d);
            let wo_o = wv_o + nk * d;
            let (wg_o, wu_o) = (wo_o + d * nq, wo_o + d * nq + nff * d);
            let wd_o = wu_o + nff * d;
            let tw = Instant::now();
            for (name, off, n_in, n_out) in [
                ("attn_q", wq_o, d, nq),
                ("attn_k", wk_o, d, nk),
                ("attn_v", wv_o, d, nk),
                ("attn_output", wo_o, nq, d),
                ("ffn_gate", wg_o, d, nff),
                ("ffn_up", wu_o, d, nff),
                ("ffn_down", wd_o, nff, d),
            ] {
                let (w, ni, no) = weight(&eng, &format!("blk.{l}.{name}.weight"))?;
                assert_eq!((ni, no), (n_in, n_out), "{name} dims");
                assert!(ni % 16 == 0);
                unsafe {
                    let b: &[u8] = std::slice::from_raw_parts(w.as_ptr().cast::<u8>(), w.len() * 4);
                    upload(&cx, stg, stg_p, stg_n, b, wbuf, (off * 4) as u64);
                }
            }
            unsafe {
                let b: &[u8] = std::slice::from_raw_parts(sbh.as_ptr().cast::<u8>(), sbh.len() * 4);
                upload(&cx, stg, stg_p, stg_n, b, sb, 0);
            }
            let t_w = tw.elapsed().as_secs_f64();

            // ---- output files ----
            let qw = if qtail.is_empty() {
                nq * tn
            } else {
                nq * 64 * qtail.len()
            };
            let open = |n: String, bytes: usize| -> R<std::fs::File> {
                let f = std::fs::OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .read(true)
                    .write(true)
                    .open(dir.join(n))?;
                if c0 == 0 {
                    f.set_len(0)?;
                    f.set_len(bytes as u64)?;
                }
                Ok(f)
            };
            let (fq, fk, fv) = (
                open(format!("q{l}.bin"), qw * 4)?,
                open(format!("k{l}.bin"), nk * tn * 4)?,
                open(format!("v{l}.bin"), nk * tn * 4)?,
            );
            let fx = std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(&xnext)?;
            if c0 == 0 {
                fx.set_len(0)?;
                fx.set_len((tn * d * 4) as u64)?;
            }
            // ---- rebuild rotated K and V of the chunks already done ----
            for c in 0..c0 {
                let r = rows(c);
                let kb = r.len() * nk * 4;
                let ks = unsafe { std::slice::from_raw_parts_mut(stg_p, 2 * kb) };
                fk.read_exact_at(&mut ks[..kb], (r.start * nk * 4) as u64)?;
                fv.read_exact_at(&mut ks[kb..], (r.start * nk * 4) as u64)?;
                unsafe {
                    setp(
                        &ropek,
                        &[
                            r.len() as u32,
                            nkv as u32,
                            half as u32,
                            r.start as u32,
                            r.start as u32,
                        ],
                    );
                    cx.run(|vk, cb| {
                        copy(vk, cb, stg, sk, 0, 0, kb as u64);
                        copy(
                            vk,
                            cb,
                            stg,
                            vd,
                            kb as u64,
                            (r.start * nk * 4) as u64,
                            kb as u64,
                        );
                        bar(vk, cb);
                        disp(
                            vk,
                            cb,
                            &ropek,
                            [((r.len() * nkv * half) as u32).div_ceil(256), 1, 1],
                        );
                    });
                }
            }
            let (mut t_a, mut t_b, mut t_c) = (0f64, 0f64, 0f64);
            for c in c0..nc {
                let r = rows(c);
                let nrow = r.len();
                let (n32, d32) = (nrow as u32, d as u32);
                let xs = unsafe { std::slice::from_raw_parts_mut(stg_p, nrow * d * 4) };
                xf.read_exact_at(xs, (r.start * d * 4) as u64)?;
                let gt = |n_out: usize| [(n_out / 128) as u32, n32.div_ceil(128), 1];
                // A: rmsnorm, q k v, rope, readback
                let kb = (nrow * nk * 4) as u64;
                let qb_off = 2 * kb;
                let tails: Vec<(usize, usize)> = qtail
                    .iter()
                    .enumerate()
                    .filter(|&(_, &t)| t - 64 >= r.start && t <= r.end)
                    .map(|(i, &t)| (i, t - 64 - r.start))
                    .collect();
                unsafe {
                    setp(&rms_a, &[d32, eps_bits, 0, n32]);
                    setp(
                        &gq,
                        &[
                            nq as u32,
                            d32,
                            n32,
                            wq_o as u32,
                            bo(o_bq, &bq),
                            0,
                            nq as u32,
                            0,
                        ],
                    );
                    setp(
                        &gk,
                        &[
                            nk as u32,
                            d32,
                            n32,
                            wk_o as u32,
                            bo(o_bk, &bkk),
                            0,
                            nk as u32,
                            0,
                        ],
                    );
                    setp(
                        &gv,
                        &[
                            nk as u32,
                            d32,
                            n32,
                            wv_o as u32,
                            bo(o_bv, &bv),
                            0,
                            nk as u32,
                            0,
                        ],
                    );
                    setp(&ropeq, &[n32, nh as u32, half as u32, r.start as u32, 0]);
                    setp(
                        &ropek,
                        &[n32, nkv as u32, half as u32, r.start as u32, r.start as u32],
                    );
                    t_a += cx.run(|vk, cb| {
                        copy(vk, cb, stg, xb, 0, 0, (nrow * d * 4) as u64);
                        bar(vk, cb);
                        disp(vk, cb, &rms_a, [n32, 1, 1]);
                        bar(vk, cb);
                        disp(vk, cb, &gq, gt(nq));
                        disp(vk, cb, &gk, gt(nk));
                        disp(vk, cb, &gv, gt(nk));
                        bar(vk, cb);
                        copy(vk, cb, sk, rb, 0, 0, kb);
                        copy(vk, cb, sv, rb, 0, kb, kb);
                        copy(vk, cb, sv, vd, 0, (r.start * nk * 4) as u64, kb);
                        if qtail.is_empty() {
                            copy(vk, cb, sq, rb, 0, qb_off, (nrow * nq * 4) as u64);
                        } else {
                            for &(i, row) in &tails {
                                copy(
                                    vk,
                                    cb,
                                    sq,
                                    rb,
                                    (row * nq * 4) as u64,
                                    qb_off + (i * 64 * nq * 4) as u64
                                        - (tails[0].0 * 64 * nq * 4) as u64,
                                    (64 * nq * 4) as u64,
                                );
                            }
                        }
                        disp(
                            vk,
                            cb,
                            &ropeq,
                            [((nrow * nh * half) as u32).div_ceil(256), 1, 1],
                        );
                        disp(
                            vk,
                            cb,
                            &ropek,
                            [((nrow * nkv * half) as u32).div_ceil(256), 1, 1],
                        );
                        bar(vk, cb);
                        host_barrier(vk, cb);
                    });
                    let rbs = std::slice::from_raw_parts(rb_p, rb_n as usize);
                    fk.write_all_at(&rbs[..kb as usize], (r.start * nk * 4) as u64)?;
                    fv.write_all_at(
                        &rbs[kb as usize..2 * kb as usize],
                        (r.start * nk * 4) as u64,
                    )?;
                    if qtail.is_empty() {
                        fq.write_all_at(
                            &rbs[qb_off as usize..qb_off as usize + nrow * nq * 4],
                            (r.start * nq * 4) as u64,
                        )?;
                    } else {
                        for (j, &(i, _)) in tails.iter().enumerate() {
                            let src = qb_off as usize + j * 64 * nq * 4;
                            fq.write_all_at(
                                &rbs[src..src + 64 * nq * 4],
                                (i * 64 * nq * 4) as u64,
                            )?;
                        }
                    }
                    // B: attention
                    setp(
                        &attn,
                        &[nh as u32, nkv as u32, r.start as u32, n32 / 64, scale_bits],
                    );
                    t_b += cx.run(|vk, cb| {
                        disp(vk, cb, &attn, [n32 / 64, nh as u32, 1]);
                        bar(vk, cb);
                    });
                    // C: Wo, rmsnorm, FFN
                    setp(&go, &[d32, nq as u32, n32, wo_o as u32, NONE, 0, d32, 1]);
                    setp(&rms_b, &[d32, eps_bits, d32, n32]);
                    setp(
                        &gg,
                        &[nff as u32, d32, n32, wg_o as u32, NONE, 0, nff as u32, 0],
                    );
                    setp(
                        &gu,
                        &[nff as u32, d32, n32, wu_o as u32, NONE, 0, nff as u32, 0],
                    );
                    setp(&silu, &[(nrow * nff / 4) as u32]);
                    setp(&gd, &[d32, nff as u32, n32, wd_o as u32, NONE, 0, d32, 1]);
                    t_c += cx.run(|vk, cb| {
                        disp(vk, cb, &go, gt(d));
                        bar(vk, cb);
                        disp(vk, cb, &rms_b, [n32, 1, 1]);
                        bar(vk, cb);
                        disp(vk, cb, &gg, gt(nff));
                        disp(vk, cb, &gu, gt(nff));
                        bar(vk, cb);
                        disp(
                            vk,
                            cb,
                            &silu,
                            [((nrow * nff / 4) as u32).div_ceil(256), 1, 1],
                        );
                        bar(vk, cb);
                        disp(vk, cb, &gd, gt(d));
                        bar(vk, cb);
                        copy(vk, cb, xb, rb, 0, 0, (nrow * d * 4) as u64);
                        host_barrier(vk, cb);
                    });
                    let rbs = std::slice::from_raw_parts(rb_p, nrow * d * 4);
                    fx.write_all_at(rbs, (r.start * d * 4) as u64)?;
                }
                if c + 1 < nc && t0.elapsed().as_secs_f64() > budget {
                    std::fs::write(&next_path, format!("{l} {}\n", c + 1))?;
                    println!(
                        "budget {budget}s reached at layer {l} chunk {}; rerun to continue",
                        c + 1
                    );
                    return Ok(ExitCode::from(3));
                }
            }
            drop((fq, fk, fv, fx));
            std::fs::rename(&xnext, &xpath)?;
            std::fs::write(&next_path, format!("{}\n", l + 1))?;
            let tl_s = tl.elapsed().as_secs_f64();
            let gem_s = t_a + t_c;
            println!(
                "layer {l}: weights {t_w:.2}s qkv {t_a:.2}s attention {t_b:.2}s wo+ffn {t_c:.2}s total {tl_s:.2}s; gemm {:.2} TFLOP/s, attention {:.2} TFLOP/s; elapsed {:.0}s",
                layer_gemm / gem_s / 1e12,
                attn_flops / t_b.max(1e-9) / 1e12,
                t0.elapsed().as_secs_f64()
            );
            if let Some(m) = vram_used_mib() {
                vram_peak = vram_peak.max(m);
            }
            if let Some(rd) = &refd {
                if layer_of(rd) > l {
                    worst = worst.max(cmp_layer(&dir, rd, l, (nq, nk), &qtail)?);
                } else {
                    println!("cmp layer {l}: reference not yet at this layer; skipped");
                }
            }
            if l + 1 < nl && t0.elapsed().as_secs_f64() > budget {
                println!("budget {budget}s reached after layer {l}; rerun to continue");
                return Ok(ExitCode::from(3));
            }
        }
        println!(
            "dump complete: {nl} layers; device allocated {:.2} GiB, nvidia-smi peak {vram_peak} MiB; host RSS peak {:.0} MiB (now {:.0} MiB)",
            cx.vram as f64 / 1073741824.0,
            rss_kb("VmHWM:") as f64 / 1024.0,
            rss_kb("VmRSS:") as f64 / 1024.0
        );
        if refd.is_some() {
            println!(
                "max rel-L2 over compared layers: {worst:.3e} (bar {BAR:.0e}): {}",
                if worst <= BAR { "PASS" } else { "FAIL" }
            );
            if worst > BAR {
                return Ok(ExitCode::FAILURE);
            }
        }
        Ok(ExitCode::SUCCESS)
    }
}
