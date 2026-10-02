//! Gate 2 for the SIMT substrate value: run all 12 ops on a real GPU and compare each result
//! against the CPU reference (`simt::op_ref`) — the run-validation the scalar substrates get
//! from exec/qemu (song `TestMetal.lean`: emit → run → compare exit code), here through the one
//! host cut a GPU has: Vulkan dispatch (`apeiron::gpu::vulkan`, dlopen only, no crate).
//!
//! Honest skip (exit 0, "SKIP") when no Vulkan loader/device is present — the `execVia` pattern.
//! Any result mismatch is a hard failure (exit 1). Objects are not destroyed on exit: the
//! process teardown reclaims them; this is a check, not a resident service.
#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "faithful port of fold's modelfold/examples/simt_check.rs; lint profile is fold's"
)]

use apeiron::gpu::*;
use apeiron::simt::{op_ref, spv_bytes, Op};
use std::process::exit;

fn main() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(msg) => {
            println!("SKIP: {msg}");
            return;
        }
    };
    unsafe { run(&vk) }
}

unsafe fn run(vk: &Vk) {
    // The {a, b, out} SSBO: 12 bytes, host-visible+coherent, mapped once.
    let Some((buffer, memory)) = vk.buffer(
        12,
        BUFFER_USAGE_STORAGE,
        MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
    ) else {
        println!("SKIP: no host-visible+coherent memory type");
        return;
    };
    let cells = vk.map(memory).cast::<u32>(); // [a, b, out]
    let (layout, dset) = vk.ssbo_set(&[buffer]);

    // Signed vectors exercising both signs, wrap, and the doorways; per-op sanitation below
    // (SDiv: no zero divisor, no MIN/-1 overflow; shift counts masked to 0..31 — SPIR-V leaves
    // shift ≥ width undefined, so the masked count is written to the GPU too, same input both sides).
    let vectors: [(i32, i32); 10] = [
        (7, 3),
        (3, 7),
        (-7, 3),
        (7, -3),
        (-7, -3),
        (0, 1),
        (-1, 1),
        (i32::MAX, 2),
        (i32::MIN, 1),
        (12345, 31),
    ];
    let mut failures = 0usize;
    for op in Op::ALL {
        let pipeline = vk.pipeline(&spv_bytes(op), layout);
        let cb = vk.command_buffer();
        vk.begin(cb);
        (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, pipeline);
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
        (vk.dispatch)(cb, 1, 1, 1);
        check((vk.end_cb)(cb), "vkEndCommandBuffer");

        let mut ran = 0usize;
        let mut ok = true;
        for &(a0, b0) in &vectors {
            let (a, b) = match op {
                Op::ADiv => {
                    let b = if b0 == 0 { 3 } else { b0 };
                    if a0 == i32::MIN && b == -1 {
                        continue; // the one overflow the ISA (and x86 #DE) excludes
                    }
                    (a0, b)
                }
                Op::AShift => (a0, b0 & 31),
                _ => (a0, b0),
            };
            let (ua, ub) = (a as u32, b as u32);
            cells.write_volatile(ua);
            cells.add(1).write_volatile(ub);
            cells.add(2).write_volatile(0xDEAD_BEEF);
            vk.submit_wait(cb);
            let got = cells.add(2).read_volatile();
            let want = op_ref(op, ua, ub);
            if got != want {
                eprintln!(
                    "FAIL {}: a={a} b={b} → gpu {got:#x}, ref {want:#x}",
                    op.name()
                );
                ok = false;
            }
            ran += 1;
        }
        if ok {
            println!("PASS {:6} ({ran} vectors)", op.name());
        } else {
            failures += 1;
        }
    }

    if failures == 0 {
        println!(
            "12/12 ops PASS on {} (SPIR-V via our own encoder + dispatch)",
            vk.device_name
        );
    } else {
        eprintln!("{failures}/12 ops FAILED on {}", vk.device_name);
        exit(1);
    }
}
