//! The obligations each face carries, checked rather than described.
//!
//! `src/keystore.rs` derives three storage faces from the three values of `sigma`, and `src/map.rs`
//! builds two of them. Those derivations are prose plus a handful of Kani harnesses over the
//! algebra itself. What this file checks is the other end: that the *table* keeps the promises the
//! algebra makes for it, on real keys, including the ones a proof cannot reach because they involve
//! `String` and an allocator.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "this is a test, not library code, and each lint below is listed because this file actually \
              triggers it: sizes and indices narrowed to build keys, bounded by the loop that produced them; \
              the reverse of that reinterpretation. Listed individually rather than suppressed wholesale, so \
              a pedantic lint this file does not yet trigger still surfaces"
)]

use apeiron::coord::Coord;
use apeiron::cell::Sigma;
use apeiron::map::{Map, Words};

/// **The `SPREAD` obligation.**
///
/// `Coord::SPREAD` says a fold is already mixed, and the parabolic face then skips `Salt::spread`
/// entirely. That is sound for *distribution* -- `mix2` mixes as well as `spread` would -- but it
/// silently moves where the **pepper** enters: from the bijection's multiplier into the fold's own
/// seed. So a type may only claim `SPREAD` if its `coord_with` actually consumes the pepper.
///
/// A type that claimed it without doing so would compile, pass every distribution test, and lose
/// the aiming defence with nothing to show for it -- the exact shape of bug this crate has now
/// shipped twice. Hence a test rather than a comment.
#[test]
fn every_already_mixed_fold_consumes_the_pepper() {
    fn obliged<T: Coord + ?Sized>(label: &str, k: &T) {
        if !T::SPREAD {
            return; // the spread supplies the pepper for these
        }
        let peppers = [0u64, 1, 0x5EED_1234_ABCD_9876, u64::MAX];
        let mut seen = std::collections::HashSet::new();
        for p in peppers {
            seen.insert(k.coord_with(p));
        }
        assert_eq!(
            seen.len(),
            peppers.len(),
            "{label} claims SPREAD but ignores the pepper, so peppering it does nothing"
        );
    }

    obliged("str", "nw_wswmg001");
    obliged("String", &String::from("nw_wswmg001"));
    obliged("[u8]", &b"nw_wswmg001"[..]);
    obliged("Vec<u8>", &b"nw_wswmg001".to_vec());
    obliged("[u8; N]", b"nw_wswmg001");
    obliged("&str", &"nw_wswmg001");
}

/// The converse obligation, and it is a **compile-time** one because it can be.
///
/// An integer's fold is the identity, so the spread is the only thing mixing it and the only thing
/// carrying the pepper. A type that wrongly claimed `SPREAD` would put every key of a
/// `Map<u32, _>` in one group and drop the pepper on the floor. A runtime assertion would catch
/// that when someone ran the test; this catches it when someone writes the line.
const _: () = assert!(
    !<u64 as Coord>::SPREAD,
    "an integer's address is the integer, so it needs the spread"
);
const _: () = assert!(!<u32 as Coord>::SPREAD);
const _: () = assert!(!<usize as Coord>::SPREAD);
const _: () = assert!(
    !<(u32, u32) as Coord>::SPREAD,
    "a tuple of integers is not peppered by its components, so it must not skip the spread"
);
// And the types that do claim it must keep claiming it -- the test above only checks the ones it
// can name, and this is what stops a refactor quietly turning the optimisation off.
const _: () = assert!(<str as Coord>::SPREAD);
const _: () = assert!(<String as Coord>::SPREAD);
const _: () = assert!(<[u8] as Coord>::SPREAD);

/// **The elliptic face regenerates what it never stored.**
///
/// `Regen` keeps only the identity word. If the bijection were not exact the key would come back
/// wrong, so this walks a real table and checks every key survives the round trip through storage.
#[test]
fn the_elliptic_face_hands_back_the_key_it_was_given() {
    let mut m: Words<u64, u32> = Words::new();
    let keys: Vec<u64> = (0..if cfg!(miri) { 200 } else { 5_000u64 })
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i << 40))
        .collect();
    for (i, &k) in keys.iter().enumerate() {
        m.insert(k, i as u32);
    }
    assert_eq!(m.face(), Sigma::Elliptic);
    let mut back: Vec<u64> = m.keys().collect();
    let mut want = keys.clone();
    back.sort_unstable();
    want.sort_unstable();
    want.dedup();
    assert_eq!(back, want, "a key came back different from the one stored");
}

/// **The parabolic face decides on the key, not the address.**
///
/// Its fold is a discard, so two distinct keys can share an identity word. When that happens the
/// table must still answer correctly -- that is the whole reason the face carries the key. Rather
/// than search for a natural collision, this forces the situation: a key type whose address is
/// deliberately lossy, so fibers are guaranteed and every lookup exercises one.
#[test]
fn the_parabolic_face_is_right_even_when_the_address_collides() {
    #[derive(PartialEq, Eq, Clone, Debug)]
    struct Lossy(u32);
    impl Coord for Lossy {
        // Deliberately terrible: only the low four bits reach the address, so 4,096 keys share
        // 16 addresses. A face that trusted the address would be wrong 4,080 times.
        fn coord(&self) -> u64 {
            u64::from(self.0 & 0xF)
        }
    }

    // `Lossy` folds to sixteen addresses on purpose, so every insert walks a fiber and the build
    // is quadratic *by design*. That is cheap at 4,096 natively and is the slowest thing in the
    // file under Miri; the fiber behaviour it checks is the same at 512.
    const LOSSY: u32 = if cfg!(miri) { 512 } else { 4_096 };
    let mut m: Map<Lossy, u32> = Map::new();
    for i in 0..LOSSY {
        m.insert(Lossy(i), i);
    }
    assert_eq!(m.len(), LOSSY as usize, "the fibers swallowed keys");
    assert_eq!(m.face(), Sigma::Parabolic);
    for i in 0..LOSSY {
        assert_eq!(m.get(&Lossy(i)), Some(&i), "wrong value for {i}");
    }
    assert_eq!(m.get(&Lossy(LOSSY + 1)), None);

    // It is right, and it is *slow*, which is the honest cost of a bad fold on this face: the
    // report says so rather than hiding it.
    let s = m.stats();
    assert!(s.looks_aimed(), "a 16-address fold should be visible in the report: {s:?}");
    for i in (0..LOSSY).step_by(7) {
        assert_eq!(m.remove(&Lossy(i)), Some(i));
    }
    for i in 0..LOSSY {
        let want = if i % 7 == 0 { None } else { Some(&i) };
        assert_eq!(m.get(&Lossy(i)), want);
    }
}

/// The two faces disagree about what a key *is* on the way out, and both are usable. This is the
/// `Held::Ref` split: a borrow where the key was stored, a fresh value where it is regenerated.
#[test]
fn each_face_hands_back_what_it_can() {
    let mut stored: Map<String, u32> = Map::new();
    stored.insert("resref".into(), 1);
    let k: &String = stored.keys().next().expect("one key");
    assert_eq!(k, "resref");

    let mut regen: Words<u64, u32> = Words::new();
    regen.insert(4_096, 1);
    let k: u64 = regen.keys().next().expect("one key");
    assert_eq!(k, 4_096);
}

/// **What each face costs, in bytes, is what the trichotomy says it should.**
///
/// The derivation in `keystore.rs` is not decoration: the elliptic face stores nothing beyond the
/// address because the address *is* the key, and the parabolic face stores the key because it is
/// not. That is a claim about memory and it is checkable as one.
#[test]
fn a_face_costs_exactly_what_its_discriminant_says() {
    use apeiron::map::{Regen, Stored};

    // Elliptic: one word, and no room for a key.
    assert_eq!(size_of::<Regen<u64>>(), size_of::<u64>());
    assert_eq!(size_of::<Regen<(u32, u32)>>(), size_of::<u64>());

    // Parabolic: one word *and* the key, both carried.
    assert_eq!(size_of::<Stored<u64>>(), size_of::<u64>() + size_of::<u64>());
    assert_eq!(size_of::<Stored<String>>(), size_of::<u64>() + size_of::<String>());

    // Which is the whole saving, stated as the difference: a `u64` key on the elliptic face costs
    // eight bytes a slot less than the same key on the parabolic one, forever.
    assert_eq!(
        size_of::<Stored<u64>>() - size_of::<Regen<u64>>(),
        size_of::<u64>(),
        "the elliptic face is supposed to save exactly the key"
    );
}

/// **One merge, on every face.**
///
/// `cell::fold`'s commutativity and associativity hold with no hypothesis on the dial, so a
/// commutative merge must give the same table whatever order the stream arrived in -- and must do
/// so identically on both keyed faces, since the merge is the half that does not vary with sigma.
///
/// The directional merges are checked to *fail* this, because a claim that some merges are
/// order-free is empty unless the others are not.
#[test]
fn a_commutative_merge_does_not_care_about_order() {
    use apeiron::merge::{count_into, set_into};

    let pairs: Vec<(u64, i64)> = (0..if cfg!(miri) { 500 } else { 2_000u64 }).map(|i| (i % 250, i as i64)).collect();
    let mut reversed = pairs.clone();
    reversed.reverse();

    // Elliptic face.
    let build = |src: &[(u64, i64)]| {
        let mut m: Words<u64, i64> = Words::new();
        for &(k, v) in src {
            m.insert_with(k, v, count_into);
        }
        m
    };
    assert_eq!(build(&pairs), build(&reversed), "counting is not order-free on the elliptic face");

    // Parabolic face, same stream, same answer.
    let build_s = |src: &[(u64, i64)]| {
        let mut m: Map<String, i64> = Map::new();
        for &(k, v) in src {
            m.insert_with(format!("k{k}"), v, count_into);
        }
        m
    };
    assert_eq!(build_s(&pairs), build_s(&reversed), "counting is not order-free on the parabolic face");

    // And the directional one is not, or the distinction means nothing.
    let build_set = |src: &[(u64, i64)]| {
        let mut m: Words<u64, i64> = Words::new();
        for &(k, v) in src {
            m.insert_with(k, v, set_into);
        }
        m
    };
    assert_ne!(
        build_set(&pairs),
        build_set(&reversed),
        "last-write-wins came out order-free, so the Species distinction is empty"
    );
}

/// A face claiming [`Sigma::key_survives`] must actually reproduce the key it was given.
///
/// The compile-time half of this is in `map.rs`: `stores_the_key` is a claim about the *width* of
/// a slot, so a `const` block decides it and a wrong constant stops the build. This is the other
/// half, which sizes cannot see -- that what comes back out is the key that went in, whether the
/// face kept it or regenerated it from the address.
///
/// Written generically over `Held` on purpose. Both keyed faces are held to the same law, and the
/// law is read off `FACE` rather than off which type it is, so it is the constant that is on trial
/// and not the implementation.
fn the_key_survives_exactly_as_the_face_claims<K, H>(keys: &[K])
where
    K: Coord + Eq + Clone + std::fmt::Debug,
    H: apeiron::map::Held<K>,
{
    for pepper in [0u64, 1, 0x5EED_1234_ABCD_9876, u64::MAX] {
        let salt = apeiron::map::Salt::new(pepper);
        for k in keys {
            let a = H::address(salt, k);
            let held = H::hold(a.id, k.clone());

            assert!(
                H::FACE.key_survives(),
                "a `Held` impl on a face that discards the key would have nothing to hand back"
            );
            assert_eq!(
                &held.into_key(salt),
                k,
                "{:?} claims the key survives, and it did not come back (pepper {pepper:#x})",
                H::FACE
            );
        }
    }
}

#[test]
fn both_keyed_faces_reproduce_the_key_they_were_given() {
    let words: Vec<u64> = (0..512u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
    // Elliptic: nothing is stored, so this is the address bijection run backwards.
    the_key_survives_exactly_as_the_face_claims::<u64, apeiron::map::Regen<u64>>(&words);
    // Parabolic: the key is carried, so this is that it is carried faithfully.
    the_key_survives_exactly_as_the_face_claims::<u64, apeiron::map::Stored<u64>>(&words);

    let names: Vec<String> = (0..256).map(|i| format!("nw_wswmg{i:05}")).collect();
    the_key_survives_exactly_as_the_face_claims::<String, apeiron::map::Stored<String>>(&names);
}
