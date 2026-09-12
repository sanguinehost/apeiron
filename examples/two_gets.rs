//! The two lookup paths, isolated for disassembly. `objdump -d` on this example's binary and read
//! `ap_get` against `hb_get`.
//!
//! Takes a map and a key, returns the value. Nothing else is in either function, so the byte
//! ranges between the two labels are the whole address-to-slot path each design pays.

use apeiron::Map;
use foldhash::fast::RandomState;
use hashbrown::HashMap as HbMap;

#[no_mangle]
#[inline(never)]
pub extern "C" fn ap_get(m: &Map<u64, u64>, k: u64) -> u64 {
    match m.get(&k) {
        Some(v) => *v,
        None => 0,
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn hb_get(m: &HbMap<u64, u64, RandomState>, k: u64) -> u64 {
    match m.get(&k) {
        Some(v) => *v,
        None => 0,
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn ap_ins(m: &mut Map<u64, u64>, k: u64) {
    m.insert(k, k);
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn hb_ins(m: &mut HbMap<u64, u64, RandomState>, k: u64) {
    m.insert(k, k);
}

// Not `extern "C"`: `&str` is not FFI-safe and the ABI is irrelevant here -- all that is wanted is
// a symbol `objdump` can find. `#[used]` keeps it from being stripped.
#[must_use]
#[export_name = "ap_str_get"]
#[inline(never)]
pub fn ap_str_get(m: &Map<String, u64>, k: &str) -> u64 {
    m.get_at(k).copied().unwrap_or(0)
}

#[must_use]
#[export_name = "hb_str_get"]
#[inline(never)]
pub fn hb_str_get(m: &HbMap<String, u64, RandomState>, k: &str) -> u64 {
    m.get(k).copied().unwrap_or(0)
}

fn main() {
    let mut a: Map<u64, u64> = Map::with_capacity(4096);
    let mut h: HbMap<u64, u64, RandomState> =
        HbMap::with_capacity_and_hasher(4096, RandomState::default());
    for i in 0..1000u64 {
        a.insert(i * 7, i);
        h.insert(i * 7, i);
    }
    let mut sm: Map<String, u64> = Map::new();
    let mut hm: HbMap<String, u64, RandomState> = HbMap::default();
    for i in 0..1000u64 {
        let k = format!("nw_wswmg{i:05}");
        sm.insert(k.clone(), i);
        hm.insert(k, i);
    }
    let mut s = 0u64;
    for i in 0..1000u64 {
        s = s
            .wrapping_add(ap_get(&a, i * 7))
            .wrapping_add(hb_get(&h, i * 7));
        let k = format!("nw_wswmg{i:05}");
        s = s
            .wrapping_add(ap_str_get(&sm, &k))
            .wrapping_add(hb_str_get(&hm, &k));
        ap_ins(&mut a, i * 13);
        hb_ins(&mut h, i * 13);
    }
    println!("{s}");
}
