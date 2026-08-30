//! The two lookup paths, isolated for disassembly. `objdump -d` on this example's binary and read
//! `ap_get` against `hb_get`.
//!
//! Takes a map and a key, returns the value. Nothing else is in either function, so the byte
//! ranges between the two labels are the whole address-to-slot path each design pays.

use apeiron::Map;
use hashbrown::HashMap as HbMap;
use foldhash::fast::RandomState;

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

fn main() {
    let mut a: Map<u64, u64> = Map::with_capacity(4096);
    let mut h: HbMap<u64, u64, RandomState> =
        HbMap::with_capacity_and_hasher(4096, RandomState::default());
    for i in 0..1000u64 {
        a.insert(i * 7, i);
        h.insert(i * 7, i);
    }
    let mut s = 0u64;
    for i in 0..1000u64 {
        s = s.wrapping_add(ap_get(&a, i * 7)).wrapping_add(hb_get(&h, i * 7));
        ap_ins(&mut a, i * 13);
        hb_ins(&mut h, i * 13);
    }
    println!("{s}");
}
