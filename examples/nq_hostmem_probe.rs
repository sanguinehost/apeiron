//! Probe: four 4 GiB imported host mappings, as the campaign's streams; optional device copy.
use apeiron::gpu::vulkan::{
    check, BufferCopy, Vk, BUFFER_USAGE_STORAGE, BUFFER_USAGE_TRANSFER_DST,
    BUFFER_USAGE_TRANSFER_SRC, MEMORY_DEVICE_LOCAL,
};

fn main() {
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let gib = 4u64 << 30;
    let touch = usize::try_from(gib).expect("4 GiB fits usize");
    let usage = BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST;
    let mut bufs = Vec::new();
    for k in 0..4 {
        let (b, _m, p) = unsafe { vk.buffer_from_anon_ptr(gib, usage) }.expect("import");
        println!("import {k}: {b} {p:?}");
        bufs.push((b, p));
    }
    for (_, p) in &bufs {
        unsafe { std::ptr::write_bytes(p.cast::<u8>(), 1, touch) };
    }
    println!("touched 16 GiB");
    std::thread::sleep(std::time::Duration::from_secs(10));
    if std::env::args().any(|a| a == "copy") {
        let dev = unsafe { vk.buffer(gib, usage, MEMORY_DEVICE_LOCAL) }
            .expect("device")
            .0;
        let cb = unsafe { vk.reusable_command_buffer() };
        let region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: gib,
        };
        for (a, _) in &bufs {
            unsafe {
                vk.reset_cb(cb);
                vk.begin(cb);
                (vk.copy_buffer)(cb, *a, dev, 1, &raw const region);
                (vk.copy_buffer)(cb, dev, *a, 1, &raw const region);
                check((vk.end_cb)(cb), "vkEndCommandBuffer");
                vk.submit_wait(cb);
            }
        }
        println!("copied 4 x 4 GiB both ways");
    }
    std::thread::sleep(std::time::Duration::from_secs(20));
    println!("done");
}
