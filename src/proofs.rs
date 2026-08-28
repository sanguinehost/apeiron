//! Bounded proofs. `cfg(kani)` only.

#[cfg(kani)]
mod harnesses {
    use crate::coord::{bucket, i21, Packed};

    /// The elliptic face's requirement: the address determines the key, over the whole domain.
    #[kani::proof]
    fn a_packed_i32_pair_round_trips() {
        let x: i32 = kani::any();
        let y: i32 = kani::any();
        assert_eq!(<(i32, i32)>::unpack((x, y).packed()), (x, y));
    }

    #[kani::proof]
    fn a_packed_u32_pair_round_trips() {
        let x: u32 = kani::any();
        let y: u32 = kani::any();
        assert_eq!(<(u32, u32)>::unpack((x, y).packed()), (x, y));
    }

    /// Injectivity directly: two distinct keys never share an address.
    #[kani::proof]
    fn packing_is_injective() {
        let a: (i32, i32) = (kani::any(), kani::any());
        let b: (i32, i32) = (kani::any(), kani::any());
        kani::assume(a != b);
        assert_ne!(a.packed(), b.packed());
    }

    /// The 21-bit coordinate refuses what will not fit, and round-trips what does.
    #[kani::proof]
    fn a_narrow_coordinate_is_exact_or_refused() {
        let v: i32 = kani::any();
        match i21::new(v) {
            Some(n) => {
                assert_eq!(n.get(), v);
                assert!(v >= i21::MIN && v <= i21::MAX);
            }
            None => assert!(v < i21::MIN || v > i21::MAX),
        }
    }

    #[kani::proof]
    fn a_packed_triple_round_trips() {
        let (x, y, z): (i32, i32, i32) = (kani::any(), kani::any(), kani::any());
        kani::assume(i21::new(x).is_some() && i21::new(y).is_some() && i21::new(z).is_some());
        let k = (
            i21::new(x).unwrap(),
            i21::new(y).unwrap(),
            i21::new(z).unwrap(),
        );
        assert_eq!(<(i21, i21, i21)>::unpack(k.packed()), k);
    }

    /// `finish` must permute, not fold: it may move a key, never merge two. If it folded,
    /// `bucket` would lose keys that `packed` had separated and the elliptic face's guarantee
    /// would end at the table's door.
    ///
    /// Proved by its steps rather than end to end. `finish` is four operations, and the whole
    /// composition against two symbolic words is two pairs of 64-bit multiplies -- the solver does
    /// not return. Each step alone does, in milliseconds, and a composition of bijections is a
    /// bijection. That is the decomposition, stated so the gap is visible: what is machine-checked
    /// is that each step permutes, and the composition is an argument on top of it.
    #[kani::proof]
    fn a_xor_shift_step_is_injective() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();
        kani::assume(a != b);
        // The two shift distances `finish` uses on the way in, and the one on the way out.
        assert_ne!(a ^ (a >> 30), b ^ (b >> 30));
        assert_ne!(a ^ (a >> 27), b ^ (b >> 27));
        assert_ne!(a ^ (a >> 31), b ^ (b >> 31));
    }

    /// Multiplication by an odd constant permutes `Z/2^64`, for both of `finish`'s multipliers
    /// and for `PHI`.
    /// Stated as the difference rather than as two products: `a * C == b * C` iff
    /// `(a - b) * C == 0`, and for odd `C` that forces `a - b == 0`. One multiply instead of two,
    /// which is the difference between deciding and not.
    #[kani::proof]
    fn multiplying_by_an_odd_constant_is_injective() {
        let d: u64 = kani::any();
        kani::assume(d != 0);
        assert_ne!(d.wrapping_mul(0xBF58_476D_1CE4_E5B9), 0);
        assert_ne!(d.wrapping_mul(0x94D0_49BB_1331_11EB), 0);
        assert_ne!(d.wrapping_mul(crate::coord::PHI), 0);
    }

    /// The SWAR group scan is byte equality, on every line and every tag.
    ///
    /// `tests/both_tag_paths_agree.rs` checks this against SSE2 and against a scalar reference on
    /// four thousand sampled lines. Sampling is the wrong instrument for a bit trick: the flaw it
    /// did catch -- a matching lane borrowing into its neighbour -- needed two specific adjacent
    /// bytes, and the next one might need three. Here the line and the tag are both symbolic, so
    /// the sixteen lanes and all 256 tags are covered at once rather than visited.
    #[kani::proof]
    fn the_swar_group_scan_is_byte_equality() {
        let line: [u8; 16] = kani::any();
        let tag: u8 = kani::any();
        let got = crate::table::scan::match_tag_swar(&line, tag);
        let mut want = 0u16;
        let mut i = 0usize;
        while i < 16 {
            if line[i] == tag {
                want |= 1 << i;
            }
            i += 1;
        }
        assert_eq!(got, want);
    }

    /// And the occupancy scan is the top bit, which is what tells an empty slot from a full one.
    #[kani::proof]
    fn the_swar_occupancy_scan_is_the_top_bit() {
        let line: [u8; 16] = kani::any();
        let got = crate::table::scan::occupied_swar(&line);
        let mut want = 0u16;
        let mut i = 0usize;
        while i < 16 {
            if line[i] & 0x80 != 0 {
                want |= 1 << i;
            }
            i += 1;
        }
        assert_eq!(got, want);
    }

    /// A bucket index is in range for its table.
    #[kani::proof]
    fn a_bucket_index_fits_its_table() {
        let address: u64 = kani::any();
        let bits: u32 = kani::any();
        kani::assume(bits >= 1 && bits <= 63);
        assert!(bucket(address, bits) < (1u64 << bits));
    }
}
