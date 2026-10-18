//! `apeiron::map::fold` behind a C ABI, so SMHasher can test it.
//!
//! SMHasher calls `hash(key, len, seed, out)` and reads 8 bytes from `out`. The seed becomes the
//! fold's pepper, which is what SMHasher varies to drive its differential and seed tests.

use std::ffi::c_void;

/// # Safety
/// `key` must point to `len` readable bytes and `out` to 8 writable ones, as SMHasher guarantees.
#[no_mangle]
pub unsafe extern "C" fn apeiron_fold(key: *const c_void, len: i32, seed: u32, out: *mut c_void) {
    let bytes = if len <= 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(key.cast::<u8>(), len as usize)
    };
    let h = apeiron::map::fold::fold_bytes_with(u64::from(seed), bytes);
    std::ptr::copy_nonoverlapping(h.to_le_bytes().as_ptr(), out.cast::<u8>(), 8);
}
