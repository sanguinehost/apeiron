use super::*;
use crate::engine::kern::limb::{gemm_q32, with_device_operands};
use crate::gpu::nn::GLSLC_ARGS;
use std::path::Path;
use std::time::Instant;

/// A fixed-seed LCG word in `[-2^bits, 2^bits)`.
fn words(len: usize, seed: u64, bits: u32) -> Vec<i64> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 11) as i64 >> (53 - bits - 1)) - (1i64 << bits)
        })
        .collect()
}

/// The committed SPIR-V is what `glslc` makes of the committed GLSL.
#[test]
fn limb_spirv_is_the_glsl() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
    let out = std::env::temp_dir().join(format!("limb_gemm_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(GLSLC_ARGS)
        .arg(dir.join("limb_gemm.comp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
    assert!(st.success(), "glslc failed: {st}");
    let fresh = std::fs::read(&out).expect("glslc output");
    let _ = std::fs::remove_file(&out);
    assert!(
        fresh == LIMB_GEMM_SPV,
        "limb_gemm.spv is stale: re-run glslc"
    );
}

/// The committed plane SPIR-V is what `glslc` makes of the committed GLSL.
#[test]
fn plane_spirv_is_the_glsl() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
    let out = std::env::temp_dir().join(format!("limb_plane_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(GLSLC_ARGS)
        .arg(dir.join("limb_plane.comp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
    assert!(st.success(), "glslc failed: {st}");
    let fresh = std::fs::read(&out).expect("glslc output");
    let _ = std::fs::remove_file(&out);
    assert!(
        fresh == plane::LIMB_PLANE_SPV,
        "limb_plane.spv is stale: re-run glslc"
    );
}

/// The device plane build equals `Plane::build` (exponents and every digit byte) at the fit's
/// operand shapes and an odd one, and flags a block exponent past 62; with the CPU wall.
#[test]
fn plane_build_matches_cpu() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    // (rows, n, bits, flag expected): W of blk.23 ffn_down (o 3584, n 1024), Z_xᵀ (r 768 rows
    // over n 1024) and its transpose over o, a tail block, and words past 2^61.
    for &(rows, n, bits, flagged) in &[
        (3584usize, 1024usize, 34u32, false),
        (768, 1024, 33, false),
        (1024, 3584, 33, false),
        (100, 70, 40, false),
        (37, 33, 5, false),
        (64, 96, 61, false),
        (64, 96, 62, true),
    ] {
        let limbs = 3u32;
        // `words` draws at most 52 bits; wider words are those shifted up.
        let x: Vec<i64> = words(
            rows * n,
            0x51ed + rows as u64 + u64::from(bits),
            bits.min(50),
        )
        .into_iter()
        .map(|w| w << bits.saturating_sub(50))
        .collect();
        let t0 = Instant::now();
        let (e_cpu, dig_cpu) =
            crate::engine::kern::limb::plane_parts(&x, rows, n, limbs).expect("cpu plane");
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
        let blocks = n.div_ceil(32);
        let dig_bytes = rows * blocks * limbs as usize * 32;
        let (e_dev, dig_dev, flag, dev_ms) = unsafe {
            let host = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
            let usage = BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_DST;
            let (bx, mx) = vk.buffer(8 * x.len() as u64, usage, host).expect("x");
            let (be, me) = vk.buffer(4 * blocks as u64, usage, host).expect("e");
            let (bd, md) = vk.buffer(dig_bytes as u64, usage, host).expect("dig");
            let (bf, mf) = vk.buffer(4, usage, host).expect("flag");
            std::ptr::copy_nonoverlapping(x.as_ptr(), vk.map(mx).cast(), x.len());
            let pb = plane::PlaneBuild::new(&vk, [bx, be, bd, bf], rows, n, limbs);
            let cb = vk.command_buffer();
            vk.begin(cb);
            pb.record(&vk, cb);
            vk.barrier(
                cb,
                STAGE_COMPUTE,
                STAGE_HOST,
                ACCESS_SHADER_WRITE,
                ACCESS_HOST_READ,
            );
            check((vk.end_cb)(cb), "vkEndCommandBuffer");
            let t1 = Instant::now();
            vk.submit_wait(cb);
            let dev_ms = t1.elapsed().as_secs_f64() * 1e3;
            (
                std::slice::from_raw_parts(vk.map(me).cast::<u32>(), blocks).to_vec(),
                std::slice::from_raw_parts(vk.map(md).cast::<i8>(), dig_bytes).to_vec(),
                *vk.map(mf).cast::<u32>(),
                dev_ms,
            )
        };
        assert_eq!(flag != 0, flagged, "range flag at {rows}x{n} bits {bits}");
        if flagged {
            continue;
        }
        assert_eq!(e_dev, e_cpu, "block exponents differ at {rows}x{n}");
        let diff = dig_dev.iter().zip(&dig_cpu).filter(|(a, b)| a != b).count();
        println!(
            "limb_plane {rows}x{n} bits {bits}: cpu {cpu_ms:.2} ms, device {dev_ms:.2} ms \
             (submit incl.), differing digits {diff} of {dig_bytes}"
        );
        assert_eq!(dig_cpu.len(), dig_bytes);
        assert_eq!(diff, 0, "digits differ at {rows}x{n}");
    }
}

/// The committed placement SPIR-V is what `glslc` makes of the committed GLSL.
#[test]
fn place_spirv_is_the_glsl() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
    let out = std::env::temp_dir().join(format!("limb_place_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(GLSLC_ARGS)
        .arg(dir.join("limb_place.comp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
    assert!(st.success(), "glslc failed: {st}");
    let fresh = std::fs::read(&out).expect("glslc output");
    let _ = std::fs::remove_file(&out);
    assert!(
        fresh == place::LIMB_PLACE_SPV,
        "limb_place.spv is stale: re-run glslc"
    );
}

/// `rows × n` words whose block `b` draws `bits(b)` bits: a fixed-seed LCG as [`words`].
fn block_words(rows: usize, n: usize, seed: u64, bits: impl Fn(usize) -> u32) -> Vec<i64> {
    let mut s = seed;
    (0..rows * n)
        .map(|i| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let b = bits((i % n) / 32);
            let w = ((s >> 11) as i64 >> (53 - b.min(50) - 1)) - (1i64 << b.min(50));
            w << b.saturating_sub(50)
        })
        .collect()
}

/// The device placement equals `host_meta` (every meta word and `Tmin_g`), its fallback flag
/// equals `host_meta`'s `None` (both of its causes: a shift past 62, and the bound at shifts in
/// 46..=62) and its range flag `Layout::placement`'s `Range`, at one and two pair groups, with
/// dead blocks, an all-dead operand pair and live entries at `T = 8`.
#[test]
fn place_matches_host_meta() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    type Bits = fn(usize) -> u32;
    // (name, n, bits of a's blocks, bits of b's blocks, expected outcome).
    let cases: [(&str, usize, Bits, Bits, &str); 8] = [
        ("fit 1024", 1024, |_| 34, |_| 33, "meta"),
        ("fit 3584, two groups", 3584, |_| 33, |_| 33, "meta"),
        (
            "dead blocks",
            1024,
            |b| if b % 3 == 0 { 2 } else { 20 + b as u32 % 11 },
            |b| if b % 5 == 0 { 3 } else { 21 + b as u32 % 9 },
            "meta",
        ),
        (
            "T = 8 edge",
            3584,
            |b| 15 + b as u32 % 5,
            |b| 15 + b as u32 % 4,
            "meta",
        ),
        ("all dead", 1024, |_| 3, |_| 3, "meta"),
        (
            "shift past 62",
            1024,
            |b| if b % 2 == 0 { 12 } else { 45 },
            |b| if b % 2 == 0 { 12 } else { 45 },
            "fallback",
        ),
        (
            "bound at shift 46..=62",
            1024,
            |b| if b % 2 == 0 { 20 } else { 45 },
            |b| if b % 2 == 0 { 20 } else { 45 },
            "fallback",
        ),
        ("range", 1024, |_| 56, |_| 56, "range"),
    ];
    let (m, p, limbs) = (40usize, 24usize, 3u32);
    let mut edge = 0usize;
    for (name, n, fa, fb, want) in cases {
        let a = block_words(m, n, 0xa11 + n as u64, fa);
        let b = block_words(p, n, 0xb22 + n as u64, fb);
        let blocks = n.div_ceil(32);
        let (ea, eb, pairs, group, exp) =
            crate::engine::kern::limb::placement_parts(&a, m, &b, p, n, limbs).expect("planes");
        // `host_meta` reads the shape, the pairs and the exponents only, never the digits.
        let host = exp.map(|exp| {
            edge += exp.iter().filter(|&&t| t == Some(8)).count();
            host_meta(&DeviceOperands {
                a_dig: &[],
                b_dig: &[],
                m,
                p,
                blocks,
                limbs,
                pairs: pairs.clone(),
                group,
                exp,
            })
        });
        let head = meta_head(&pairs, group, blocks, limbs, m, p, 1);
        let (groups, segs) = segments(&pairs, group);
        let (ngroups, nsegs) = (groups.len(), segs.len());
        let meta_words = head.len() + nsegs * blocks;
        let (meta_dev, out_dev, dev_ms) = unsafe {
            let host_mem = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
            let usage = BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_DST;
            let (bea, mea) = vk.buffer(4 * blocks as u64, usage, host_mem).expect("ea");
            let (beb, meb) = vk.buffer(4 * blocks as u64, usage, host_mem).expect("eb");
            let (bm, mm) = vk
                .buffer(4 * meta_words as u64, usage, host_mem)
                .expect("meta");
            let (bo, mo) = vk
                .buffer(4 * (2 + ngroups) as u64, usage, host_mem)
                .expect("out");
            std::ptr::copy_nonoverlapping(ea.as_ptr(), vk.map(mea).cast(), blocks);
            std::ptr::copy_nonoverlapping(eb.as_ptr(), vk.map(meb).cast(), blocks);
            // The shift table and the outputs start as a pattern the kernel must overwrite.
            let pm = vk.map(mm).cast::<u32>();
            std::ptr::write_bytes(pm, 0xA5, meta_words);
            std::ptr::copy_nonoverlapping(head.as_ptr(), pm, head.len());
            std::ptr::write_bytes(vk.map(mo).cast::<u32>(), 0xA5, 2 + ngroups);
            let pl = place::Place::new(&vk, [bea, beb, bm, bo], ngroups as u32);
            let cb = vk.command_buffer();
            vk.begin(cb);
            pl.record(&vk, cb);
            vk.barrier(
                cb,
                STAGE_COMPUTE,
                STAGE_HOST,
                ACCESS_SHADER_WRITE,
                ACCESS_HOST_READ,
            );
            check((vk.end_cb)(cb), "vkEndCommandBuffer");
            let t1 = Instant::now();
            vk.submit_wait(cb);
            let dev_ms = t1.elapsed().as_secs_f64() * 1e3;
            (
                std::slice::from_raw_parts(pm, meta_words).to_vec(),
                std::slice::from_raw_parts(vk.map(mo).cast::<u32>(), 2 + ngroups).to_vec(),
                dev_ms,
            )
        };
        let (range, fallback) = (out_dev[0] != 0, out_dev[1] != 0);
        let got = match &host {
            Err(_) => "range",
            Ok(None) => "fallback",
            Ok(Some(_)) => "meta",
        };
        assert_eq!(got, want, "{name}: the host outcome is not the case's");
        assert_eq!(range, got == "range", "{name}: range flag");
        if got != "range" {
            assert_eq!(fallback, got == "fallback", "{name}: fallback flag");
        }
        let mut diff = 0usize;
        if let Ok(Some((meta, tmin))) = &host {
            diff = meta.iter().zip(&meta_dev).filter(|(x, y)| x != y).count()
                + tmin
                    .iter()
                    .zip(&out_dev[2..])
                    .filter(|(x, y)| x != y)
                    .count();
            assert_eq!(meta.len(), meta_dev.len());
        }
        println!(
            "limb_place {name}: n {n}, groups {ngroups}, segments {nsegs}, outcome {got}, \
             differing words {diff}, device {dev_ms:.3} ms (submit incl.)"
        );
        assert_eq!(diff, 0, "{name}: meta or Tmin differ");
    }
    println!("limb_place: {edge} live block pairs at T = 8");
    assert!(
        edge > 0,
        "no live block pair at T = 8: the MIN_PAIR_EXP edge is untested"
    );
}

/// The committed SPIR-V is what `glslc` makes of the committed GLSL.
#[test]
fn finish_spirv_is_the_glsl() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
    let out = std::env::temp_dir().join(format!("limb_finish_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(GLSLC_ARGS)
        .arg(dir.join("limb_finish.comp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
    assert!(st.success(), "glslc failed: {st}");
    let fresh = std::fs::read(&out).expect("glslc output");
    let _ = std::fs::remove_file(&out);
    assert!(
        fresh == finish::LIMB_FINISH_SPV,
        "limb_finish.spv is stale: re-run glslc"
    );
}

/// `limb_finish.comp` over host-visible copies of `back` (the lo/hi planes), the meta head and
/// `tmin`: the `plane` products, the range flag and the submit wall in ms.
fn finish_on_device(
    vk: &Vk,
    back: &[i32],
    meta: &[u32],
    tmin: &[u32],
    plane: usize,
) -> (Vec<i64>, bool, f64) {
    unsafe {
        let host_mem = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
        let usage = BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_DST;
        let (bo, mo) = vk
            .buffer(4 * back.len() as u64, usage, host_mem)
            .expect("o");
        let (bm, mm) = vk
            .buffer(4 * meta.len() as u64, usage, host_mem)
            .expect("meta");
        let (bp, mp) = vk
            .buffer(4 * (2 + tmin.len()) as u64, usage, host_mem)
            .expect("place");
        let (bx, mx) = vk.buffer(8 * plane as u64, usage, host_mem).expect("x");
        std::ptr::copy_nonoverlapping(back.as_ptr(), vk.map(mo).cast(), back.len());
        std::ptr::copy_nonoverlapping(meta.as_ptr(), vk.map(mm).cast(), meta.len());
        let pp = vk.map(mp).cast::<u32>();
        std::ptr::write_bytes(pp, 0, 2);
        std::ptr::copy_nonoverlapping(tmin.as_ptr(), pp.add(2), tmin.len());
        // The products start as a pattern the kernel must overwrite.
        std::ptr::write_bytes(vk.map(mx).cast::<i64>(), 0xA5, plane);
        let fin = finish::Finish::new(vk, [bo, bm, bp, bx], plane);
        let cb = vk.command_buffer();
        vk.begin(cb);
        fin.record(vk, cb);
        vk.barrier(
            cb,
            STAGE_COMPUTE,
            STAGE_HOST,
            ACCESS_SHADER_WRITE,
            ACCESS_HOST_READ,
        );
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        let t1 = Instant::now();
        vk.submit_wait(cb);
        let ms = t1.elapsed().as_secs_f64() * 1e3;
        (
            std::slice::from_raw_parts(vk.map(mx).cast::<i64>(), plane).to_vec(),
            *pp != 0,
            ms,
        )
    }
}

/// The device finish equals the product: through the device product at the fit's `W·Z` shape,
/// a Gram-like shape and an odd one (against `gemm_q32`), and on synthetic lo/hi planes that
/// reach every branch (against [`FinishTask`]): rounding at `Tmin` 0, 1, 17, 31, exact shifts at
/// 32, 50, 91, a sum that leaves `i64` and returns, a group word and a sum that do not return.
#[test]
fn finish_matches_gemm_and_host() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    if !vk.has_coopmat {
        println!("SKIP: no cooperative matrices");
        return;
    }
    let mut dev = LimbGemm::default();
    for &(m, p, n, ba, bb) in &[
        (1024usize, 768usize, 3584usize, 34u32, 33u32),
        (768, 768, 1024, 33, 33),
        (100, 37, 70, 40, 20),
    ] {
        let a = words(m * n, 0x9e37 + m as u64, ba);
        let b = words(p * n, 0x7f4a + p as u64, bb);
        let (cpu, (x, range, ms, ngroups, pp)) = with_device_operands(&a, m, &b, p, n, 3, |ops| {
            let host = unsafe { dev.run(&vk, ops) }
                .expect("device product")
                .expect("accumulator bound holds");
            let (meta, tmin) = host_meta(ops).expect("accumulator bound holds");
            let (mp, pp) = (m.div_ceil(TILE) * TILE, p.div_ceil(TILE) * TILE);
            let plane = mp * pp;
            let bound = dev.bound.as_ref().expect("bound buffers");
            let back = unsafe {
                std::slice::from_raw_parts(bound.back_ptr.cast::<i32>(), 2 * tmin.len() * plane)
            };
            let (x, range, ms) = finish_on_device(&vk, back, &meta, &tmin, plane);
            let x: Vec<i64> = (0..m * p).map(|k| x[(k / p) * pp + k % p]).collect();
            assert!(
                x == host,
                "device finish differs from the host finish at {m}x{p}x{n}"
            );
            (x, range, ms, tmin.len(), pp)
        })
        .expect("planes");
        let diff = cpu.iter().zip(&x).filter(|(u, v)| u != v).count();
        println!(
            "limb_finish {m}x{p}x{n}: groups {ngroups}, row stride {pp}, range {range}, \
             differing entries {diff} of {}, device {ms:.3} ms (submit incl.)",
            m * p
        );
        assert!(!range, "range flag at {m}x{p}x{n}");
        assert_eq!(
            diff, 0,
            "device finish differs from gemm_q32 at {m}x{p}x{n}"
        );
    }
    // Synthetic planes, 64 × 64: (name, Tmin per group, v per group and entry, flagged).
    let plane = 64 * 64;
    let lcg = |seed: u64, bits: u32| block_words(1, plane, seed, |_| bits);
    let fill = |c: i64| vec![c; plane];
    let ramp = |c: i64| (0..plane).map(|k| c - k as i64).collect::<Vec<_>>();
    let cases: Vec<(&str, Vec<u32>, Vec<Vec<i64>>, bool)> = vec![
        (
            "round at Tmin 0, 1, 17, 31",
            vec![0, 1, 17, 31],
            vec![lcg(1, 62), lcg(2, 61), lcg(3, 45), lcg(4, 31)],
            false,
        ),
        (
            "exact at Tmin 32, 50, 91",
            vec![32, 50, 91],
            vec![lcg(5, 60), lcg(6, 40), lcg(7, 2)],
            false,
        ),
        (
            "sum leaves i64 and returns",
            vec![32, 32, 32],
            // Two groups sum to 2^63 + 2·(4096 − k) > i64::MAX; the third returns it.
            vec![
                ramp((1 << 62) + plane as i64),
                ramp((1 << 62) + plane as i64),
                fill(-(1 << 62)),
            ],
            false,
        ),
        ("group word leaves i64", vec![40], vec![lcg(8, 60)], true),
        (
            "sum leaves i64",
            vec![32, 32],
            vec![fill(1 << 62), fill(1 << 62)],
            true,
        ),
    ];
    let mut rounded = 0usize;
    for (name, tmin, vs, flagged) in cases {
        let g = tmin.len();
        let mut back = vec![0i32; 2 * g * plane];
        for (k, v) in vs.iter().enumerate() {
            for (at, &w) in v.iter().enumerate() {
                back[2 * k * plane + at] = w as i32;
                back[(2 * k + 1) * plane + at] = (w >> 32) as i32;
                if tmin[k] < 32 && (i128::from(w) << tmin[k]) & (1 << 31) != 0 {
                    rounded += 1;
                }
            }
        }
        let mut meta = vec![0u32, 0, 64, 64, 9, 9 + 2 * g as u32, 0, 0, 1];
        meta.extend(std::iter::repeat_n(0, 2 * g));
        let task = FinishTask {
            back: &back,
            tmin: &tmin,
            p: 64,
            pp: 64,
            plane,
            range: AtomicBool::new(false),
        };
        let mut host = vec![0i64; plane];
        task.run(0, 0, &mut host);
        let host_range = task.range.load(Ordering::Relaxed);
        let (x, range, ms) = finish_on_device(&vk, &back, &meta, &tmin, plane);
        let diff = if host_range {
            0
        } else {
            host.iter().zip(&x).filter(|(u, v)| u != v).count()
        };
        println!(
            "limb_finish {name}: groups {g}, range host {host_range} device {range}, \
             differing entries {diff} of {plane}, device {ms:.3} ms (submit incl.)"
        );
        assert_eq!(
            host_range, flagged,
            "{name}: the host outcome is not the case's"
        );
        assert_eq!(range, host_range, "{name}: range flag");
        assert_eq!(
            diff, 0,
            "{name}: device finish differs from the host finish"
        );
    }
    println!("limb_finish: {rounded} synthetic entries with the 2^31 rounding bit set");
    assert!(rounded > 0, "no entry reaches the rounding constant");
}

/// Bit identity with the CPU product at the fit's shapes and an odd one, directly and through
/// the routed `gemm_q32`, plus the wall of the CPU kernel and of the routed call.
#[test]
fn limb_gemm_matches_cpu() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    if !vk.has_coopmat {
        println!("SKIP: no cooperative matrices");
        return;
    }
    let mut dev = LimbGemm::default();
    // (m, p, n, bits of a, bits of b): the W·Z product of blk.23 ffn_down at r 768, its Gram
    // ZᵀZ, the sketched tune's V·Qᵀ (s 128) and U·Pᵀ (inner s), the ridge solve's Rᵀ·H⁻¹, the
    // Gram-like 768² over 1024, a mid shape and an odd shape below one tile.
    for &(m, p, n, ba, bb) in &[
        (1024usize, 768usize, 3584usize, 34u32, 33u32),
        (768, 768, 3584, 33, 33),
        (768, 128, 3584, 33, 33),
        (1024, 768, 128, 33, 33),
        (1024, 768, 768, 33, 33),
        (768, 768, 1024, 33, 33),
        (256, 256, 1024, 33, 33),
        (100, 37, 70, 40, 20),
    ] {
        let a = words(m * n, 0x9e37 + m as u64, ba);
        let b = words(p * n, 0x7f4a + p as u64, bb);
        let (cpu, direct) = with_device_operands(&a, m, &b, p, n, 3, |ops| {
            let first = unsafe { dev.run(&vk, ops) }
                .expect("device product")
                .expect("accumulator bound holds");
            let again = unsafe { dev.run(&vk, ops) }
                .expect("device product")
                .expect("accumulator bound holds");
            assert!(again == first, "device product is not deterministic");
            first
        })
        .expect("planes");
        let diff = cpu.iter().zip(&direct).filter(|(x, y)| x != y).count();
        assert_eq!(diff, 0, "device differs from gemm_q32 at {m}x{p}x{n}");
        let mut cpu_ms = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = with_device_operands(&a, m, &b, p, n, 3, |_| ()).expect("cpu product");
            cpu_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let routed = gemm_q32(&a, m, &b, p, n, 3).expect("routed product");
        assert!(routed == cpu, "routed gemm_q32 differs at {m}x{p}x{n}");
        let mut routed_ms = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = gemm_q32(&a, m, &b, p, n, 3).expect("routed product");
            routed_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "limb_gemm {m}x{p}x{n}: cpu {:.2}/{:.2}/{:.2} ms, routed {:.2}/{:.2}/{:.2} ms \
             (device taken: {}), differing entries {diff} of {}",
            cpu_ms[0],
            cpu_ms[1],
            cpu_ms[2],
            routed_ms[0],
            routed_ms[1],
            routed_ms[2],
            takes(m, p, n),
            cpu.len()
        );
    }
}

/// Binary64 `/`, `sqrt`, unfused `a·b + c`, round-half-away of
/// `a·2³²` and `i64 → f64` on the device against the CPU, bit for bit.
#[test]
fn f64_maps_match_cpu() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    if !vk.has_f64 {
        println!("SKIP: no shaderFloat64");
        return;
    }
    let src = r#"#version 450
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer I { double x[]; };
layout(std430, binding = 1) writeonly buffer O { uint64_t y[]; };
layout(std430, binding = 2) readonly buffer M { uint n; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    double a = x[3 * i], b = x[3 * i + 1], c = x[3 * i + 2];
    precise double q = a / b;
    precise double s = sqrt(abs(a));
    precise double p = a * b;
    precise double m = p + c;
    precise double sc = a * 4294967296.0;
    double t = trunc(sc);
    precise double fr = abs(sc - t);
    double r = fr >= 0.5 ? t + sign(sc) : t;
    int64_t ri = int64_t(r);
    double back = double(int64_t(doubleBitsToUint64(c) >> 1) - int64_t(0x2000000000000000ul));
    y[6 * i] = doubleBitsToUint64(q);
    y[6 * i + 1] = doubleBitsToUint64(s);
    y[6 * i + 2] = doubleBitsToUint64(p);
    y[6 * i + 3] = doubleBitsToUint64(m);
    y[6 * i + 4] = uint64_t(ri);
    y[6 * i + 5] = doubleBitsToUint64(back);
}
"#;
    let dir = std::env::temp_dir();
    let comp = dir.join(format!("f64probe_{}.comp", std::process::id()));
    let spv = dir.join(format!("f64probe_{}.spv", std::process::id()));
    std::fs::write(&comp, src).expect("write probe");
    let st = std::process::Command::new("glslc")
        .args(GLSLC_ARGS)
        .arg(&comp)
        .arg("-o")
        .arg(&spv)
        .status()
        .expect("glslc");
    assert!(st.success(), "glslc failed");
    let code = std::fs::read(&spv).expect("spv");
    let n = 1usize << 20;
    let mut st = 0x2545f4914f6cdd1du64;
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    let mut val = |k: u64| -> f64 {
        let r = next();
        let mag = ((r >> 11) as f64 / (1u64 << 53) as f64) * 2f64.powi(((r & 63) as i32) - 32);
        let v = if k % 7 == 0 {
            // a·2³² on a half: k + 0.5 over 2³².
            ((r >> 34) as f64 + 0.5) / 4294967296.0
        } else {
            mag
        };
        if r & 64 != 0 {
            -v
        } else {
            v
        }
    };
    let mut x = vec![0f64; 3 * n];
    for (k, e) in x.iter_mut().enumerate() {
        *e = val(k as u64);
        if k % 3 == 1 && *e == 0.0 {
            *e = 1.0;
        }
    }
    let want: Vec<u64> = (0..n)
        .flat_map(|i| {
            let (a, b, c) = (x[3 * i], x[3 * i + 1], x[3 * i + 2]);
            let sc = a * 4294967296.0;
            let back = ((c.to_bits() >> 1) as i64 - 0x2000000000000000i64) as f64;
            [
                (a / b).to_bits(),
                a.abs().sqrt().to_bits(),
                (a * b).to_bits(),
                (a * b + c).to_bits(),
                sc.round() as i64 as u64,
                back.to_bits(),
            ]
        })
        .collect();
    let got = unsafe {
        let host = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
        let (bi, mi) = vk
            .buffer((24 * n) as u64, BUFFER_USAGE_STORAGE, host)
            .expect("in");
        let (bo, mo) = vk
            .buffer((48 * n) as u64, BUFFER_USAGE_STORAGE, host)
            .expect("out");
        let (bm, mm) = vk.buffer(16, BUFFER_USAGE_STORAGE, host).expect("meta");
        std::ptr::copy_nonoverlapping(x.as_ptr().cast::<u8>(), vk.map(mi).cast(), 24 * n);
        *vk.map(mm).cast::<u32>() = n as u32;
        let (layout, dset) = vk.ssbo_set(&[bi, bo, bm]);
        let pipe = vk.pipeline(&code, layout);
        let cb = vk.command_buffer();
        vk.begin(cb);
        (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, pipe);
        (vk.bind_sets)(
            cb,
            PIPELINE_BIND_POINT_COMPUTE,
            layout,
            0,
            1,
            &dset,
            0,
            std::ptr::null(),
        );
        (vk.dispatch)(cb, (n / 64) as u32, 1, 1);
        vk.barrier(
            cb,
            STAGE_COMPUTE,
            STAGE_HOST,
            ACCESS_SHADER_WRITE,
            ACCESS_HOST_READ,
        );
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);
        std::slice::from_raw_parts(vk.map(mo).cast::<u64>(), 6 * n).to_vec()
    };
    let _ = std::fs::remove_file(&comp);
    let _ = std::fs::remove_file(&spv);
    for (op, name) in ["div", "sqrt", "mul", "mul+add", "round", "i64->f64"]
        .iter()
        .enumerate()
    {
        let bad: Vec<usize> = (0..n)
            .filter(|&i| got[6 * i + op] != want[6 * i + op])
            .collect();
        println!("f64 {name}: {} of {n} differ", bad.len());
        for &i in bad.iter().take(3) {
            println!(
                "  a {:e} b {:e} c {:e}: device {:#x} cpu {:#x}",
                x[3 * i],
                x[3 * i + 1],
                x[3 * i + 2],
                got[6 * i + op],
                want[6 * i + op]
            );
        }
        assert!(
            bad.is_empty(),
            "device binary64 {name} differs from the CPU"
        );
    }
}

/// The committed ADMM map SPIR-V is what `glslc` makes of the committed GLSL.
#[test]
fn admm_map_spirv_is_the_glsl() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
    let out = std::env::temp_dir().join(format!("admm_map_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(GLSLC_ARGS)
        .arg(dir.join("admm_map.comp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
    assert!(st.success(), "glslc failed: {st}");
    let fresh = std::fs::read(&out).expect("glslc output");
    let _ = std::fs::remove_file(&out);
    assert!(
        fresh == admm::ADMM_MAP_SPV,
        "admm_map.spv is stale: re-run glslc"
    );
}

/// A fixed-seed LCG value in `[-1, 1)`.
fn unit(len: usize, seed: u64) -> Vec<f64> {
    words(len, seed, 30)
        .into_iter()
        .map(|w| w as f64 / f64::from(1u32 << 30))
        .collect()
}

/// The resident `x_update` equals `learn::nanoquant::x_update` in every bit at two odd shapes and
/// both sides of the blk.23 `ffn_down` frame, twice on one frame; with both walls. A nonzero cutoff
/// runs the device inverse above it (odd splits at rank 75) and the host sees only leaves.
#[test]
fn admm_x_update_matches_cpu() {
    use crate::learn::nanoquant::{spd_inverse, x_update, LIMBS};
    if device().is_none() {
        println!("SKIP: No Vulkan device with cooperative matrices");
        return;
    }
    // (k, n, r, cut): two odd shapes, side A (W, o 3584 × n 1024) and side B (Wᵀ) at rank 768.
    for &(k, n, r, cut) in &[
        (200, 96, 64, 0),
        (130, 70, 40, 0),
        (3584, 1024, 768, 0),
        (1024, 3584, 768, 0),
        (200, 96, 64, 32),
        (130, 70, 75, 32),
        (3584, 1024, 768, 96),
        (1024, 3584, 768, 192),
    ] {
        // W_n on the grid at |w| < 2^-3, as the normalised target.
        let w = words(k * n, 0x5eed ^ (k * n) as u64, 29);
        let wp = crate::engine::kern::limb::prepare(&w, k, n, LIMBS).expect("plane");
        let Some(mut side) = Side::new(&w, k, n, r, LIMBS, cut) else {
            println!("SKIP: no binary64 device frame");
            return;
        };
        // The largest order the host inverts: `r` on the host route, a leaf on the device route.
        let top = std::cell::Cell::new(0usize);
        let inv = |h: &[i64], o: usize| {
            top.set(top.get().max(o));
            spd_inverse(h, o)
        };
        for round in 0..2u64 {
            let zx = unit(n * r, 11 + round);
            let z = unit(k * r, 23 + round);
            let du = unit(k * r, 37 + round);
            let (rho, reg) = (0.3 + 0.1 * round as f64, 3e-2);
            let t = Instant::now();
            let cpu = x_update(&zx, n, r, &wp, k, &z, &du, rho, reg).expect("cpu x_update");
            let cpu_s = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let dev = side
                .x_update(&zx, &z, &du, rho, reg, inv)
                .expect("device stages flagged")
                .expect("host inverse");
            let dev_s = t.elapsed().as_secs_f64();
            let diff = cpu
                .iter()
                .zip(&dev)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            println!(
                "x_update k {k} n {n} r {r} cut {cut} round {round}: cpu {:.2} ms, device {:.2} ms, host order {}, {diff} words differ",
                cpu_s * 1e3,
                dev_s * 1e3,
                top.get()
            );
            assert_eq!(
                diff, 0,
                "x_update k {k} n {n} r {r} cut {cut} round {round}"
            );
            let want = if cut == 0 { r } else { cut };
            assert!(
                top.get() <= want,
                "host inverted order {} past {want}",
                top.get()
            );
            top.set(0);
        }
    }
}

/// The committed inverse map SPIR-V is what `glslc` makes of the committed GLSL.
#[test]
fn inv_map_spirv_is_the_glsl() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
    let out = std::env::temp_dir().join(format!("inv_map_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(GLSLC_ARGS)
        .arg(dir.join("inv_map.comp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
    assert!(st.success(), "glslc failed: {st}");
    let fresh = std::fs::read(&out).expect("glslc output");
    let _ = std::fs::remove_file(&out);
    assert!(
        fresh == inverse::INV_MAP_SPV,
        "inv_map.spv is stale: re-run glslc"
    );
}

/// The committed rank1 map SPIR-V is what `glslc` makes of the committed GLSL.
#[test]
fn rank1_map_spirv_is_the_glsl() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
    let out = std::env::temp_dir().join(format!("rank1_map_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(GLSLC_ARGS)
        .arg(dir.join("rank1_map.comp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
    assert!(st.success(), "glslc failed: {st}");
    let fresh = std::fs::read(&out).expect("glslc output");
    let _ = std::fs::remove_file(&out);
    assert!(
        fresh == rank1::RANK1_MAP_SPV,
        "rank1_map.spv is stale: re-run glslc"
    );
}

/// The resident `rank1` equals `learn::nanoquant::rank1` in every bit on an odd shape (both start
/// sides) and on both sides of the blk.23 `ffn_down` frame, three rounds on one frame (two
/// recorded `iters`); with both walls.
#[test]
fn admm_rank1_matches_cpu() {
    use crate::learn::nanoquant::{rank1, LIMBS};
    if device().is_none() {
        println!("SKIP: No Vulkan device with cooperative matrices");
        return;
    }
    // (k, n, r, start_rows): the matrix is k × r on the frame of (k, n, r).
    for &(k, n, r, sr) in &[
        (130, 70, 40, false),
        (130, 70, 40, true),
        (3584, 1024, 768, false),
        (1024, 3584, 768, true),
    ] {
        let w = words(k * n, 0x5eed ^ (k * n) as u64, 29);
        let Some(mut side) = Side::new(&w, k, n, r, LIMBS, 0) else {
            println!("SKIP: no binary64 device frame");
            return;
        };
        for (round, iters) in [(0u64, 5usize), (1, 5), (2, 1)] {
            let m = unit(k * r, 41 + round);
            let start = unit(if sr { k } else { r }, 53 + round);
            let t = Instant::now();
            let cpu = rank1(&m, k, r, iters, sr, start.clone()).expect("cpu rank1");
            let cpu_s = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let dev = side
                .rank1(&m, iters, sr, &start)
                .expect("device stages flagged");
            let dev_s = t.elapsed().as_secs_f64();
            let diff = cpu
                .iter()
                .zip(&dev)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            println!(
                "rank1 {k}x{r} start_rows {sr} iters {iters}: cpu {:.2} ms, device {:.2} ms, {diff} words differ",
                cpu_s * 1e3,
                dev_s * 1e3
            );
            assert_eq!(
                (diff, dev.len()),
                (0, cpu.len()),
                "rank1 {k}x{r} {sr} {iters}"
            );
        }
    }
}

/// Unit costs of a resident Schur inverse on the device (microbench; prints, asserts nothing): the
/// wall of one empty submit, and of one plane-build pair plus one limb product per half-order `s`
/// of the order-768 recursion, recorded `reps` times in one command buffer.
#[test]
#[ignore = "microbench"]
fn device_unit_costs() {
    use admm::{dev_buf, full, Product};
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    if !vk.has_coopmat {
        println!("SKIP: no cooperative matrices");
        return;
    }
    let limbs = 3u32;
    unsafe {
        let cb = vk.command_buffer();
        vk.begin(cb);
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);
        let t = Instant::now();
        for _ in 0..200 {
            vk.submit_wait(cb);
        }
        println!(
            "UNIT empty submit_wait {:.1} us",
            t.elapsed().as_secs_f64() * 1e6 / 200.0
        );
        for &s in &[24usize, 48, 96, 192, 384] {
            let reps = 20usize;
            let sp = s.div_ceil(TILE) * TILE;
            let blocks = s.div_ceil(32);
            let lb = limbs as usize * 32;
            let host = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
            let x = words(s * s, 0x1234 + s as u64, 33);
            let (bx, mx) = vk
                .buffer(8 * x.len() as u64, BUFFER_USAGE_TRANSFER_SRC, host)
                .expect("x");
            std::ptr::copy_nonoverlapping(x.as_ptr(), vk.map(mx).cast(), x.len());
            // The words device-local, as the resident inverse holds them; a host-visible buffer
            // read by both plane builds every rep would time PCIe, not the device.
            let dx = dev_buf(&vk, 8 * x.len()).expect("x dev");
            let pl = [
                dx,
                dev_buf(&vk, 4 * blocks).expect("e"),
                dev_buf(&vk, sp * blocks * lb).expect("dig"),
                dev_buf(&vk, 4).expect("flag"),
            ];
            let pb = plane::PlaneBuild::new(&vk, pl, s, s, limbs);
            let o = dev_buf(&vk, Product::o_bytes(s, s, s, limbs)).expect("o");
            let xo = dev_buf(&vk, 8 * sp * sp).expect("x out");
            let mut ups = Vec::new();
            let pr = Product::new(
                &vk,
                [pl[2], pl[1]],
                [pl[2], pl[1]],
                (s, s, s, limbs),
                o,
                xo,
                &mut ups,
            )
            .expect("product");
            let total: usize = ups.iter().map(|u: &(Handle, Vec<u8>)| u.1.len()).sum();
            let (st, sm) = vk
                .buffer(total.max(16) as u64, BUFFER_USAGE_TRANSFER_SRC, host)
                .expect("stage");
            let sp_ptr = vk.map(sm).cast::<u8>();
            let cb = vk.command_buffer();
            vk.begin(cb);
            (vk.fill_buffer)(cb, pl[2], 0, WHOLE_SIZE, 0);
            admm::copy(&vk, cb, bx, 0, dx, 0, 8 * x.len());
            let mut at = 0usize;
            for (dst, b) in &ups {
                std::ptr::copy_nonoverlapping(b.as_ptr(), sp_ptr.add(at), b.len());
                admm::copy(&vk, cb, st, at, *dst, 0, b.len());
                at += b.len();
            }
            check((vk.end_cb)(cb), "vkEndCommandBuffer");
            vk.submit_wait(cb);
            for (tag, planes, product) in [
                ("two planes + product", true, true),
                ("two planes", true, false),
                ("product", false, true),
            ] {
                let cb = vk.command_buffer();
                vk.begin(cb);
                for _ in 0..reps {
                    if planes {
                        pb.record(&vk, cb);
                        full(&vk, cb);
                        pb.record(&vk, cb);
                        full(&vk, cb);
                    }
                    if product {
                        pr.record(&vk, cb);
                        full(&vk, cb);
                    }
                }
                check((vk.end_cb)(cb), "vkEndCommandBuffer");
                vk.submit_wait(cb);
                let mut us = Vec::new();
                for _ in 0..5 {
                    let t = Instant::now();
                    vk.submit_wait(cb);
                    us.push(t.elapsed().as_secs_f64() * 1e6 / reps as f64);
                }
                let mean = us.iter().sum::<f64>() / us.len() as f64;
                println!("UNIT s {s} {tag} {mean:.1} us per rep (reps {reps}, 5 runs)");
            }
        }
    }
}
