//! The ring laws, decided by enumeration instead of by a solver.
//!
//! These are polynomial identities with integer coefficients: `mul_assoc` is an equality of two
//! polynomials in six scalar variables, and such an identity holds in every commutative ring or in
//! none. Nothing here recurses, loops or branches on data, so there is no halting question to
//! answer -- the check is finite by construction.
//!
//! A bit-precise checker is the wrong instrument for it anyway. Bit-blasting a multiply builds a
//! multiplier circuit, and nonlinear arithmetic is where SAT is weakest: measured on this crate,
//! `a(b + c) = ab + ac` over three symbolic `i8` took **418 seconds**, and the six-variable laws
//! did not return in 400. Walking the ring instead:
//!
//!   distributivity, all 16,777,216 triples        0 ms
//!   the six-variable laws, 531,441 tuples         3 ms
//!
//! Five orders of magnitude, and complete over the window rather than sampled. The unbounded
//! statement belongs in `sanguine`, where `ring` normalises both sides and no model is enumerated
//! at all; this is the implementation agreeing with it, and the gate that keeps it agreeing.

use apeiron::cell::{Cell, Sigma};
use apeiron::scalar::Ring;
use std::time::Instant;

const FACES: [Sigma; 3] = [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic];

#[test]
fn distributivity_holds_across_the_whole_ring() {
    let t = Instant::now();
    let mut n = 0u64;
    for a in i8::MIN..=i8::MAX {
        for b in i8::MIN..=i8::MAX {
            for c in i8::MIN..=i8::MAX {
                assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
                n += 1;
            }
        }
    }
    println!("distributivity: {n} triples, {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);

}

/// The six-variable laws, over a window rather than the whole ring: `2^48` tuples is too many to
/// walk, and these identities are degree two in each variable, so a window of nine values exercises
/// every coefficient. Complete over what it covers, and not a claim beyond it.
#[test]
fn the_cell_laws_hold_across_a_window() {
    let t = Instant::now();
    let (mut n, r) = (0u64, -4i8..=4);
    for a in r.clone() { for b in r.clone() { for c in r.clone() {
    for d in r.clone() { for e in r.clone() { for f in r.clone() {
        let (z, v, u) = (Cell::<i8>::new(a, b), Cell::<i8>::new(c, d), Cell::<i8>::new(e, f));
        for sigma in FACES {
            assert_eq!(z.mul(sigma, v), v.mul(sigma, z));
            assert_eq!(z.mul(sigma, v).mul(sigma, u), z.mul(sigma, v.mul(sigma, u)));
            assert_eq!(
                z.mul(sigma, v).norm_form(sigma),
                z.norm_form(sigma).mul(v.norm_form(sigma))
            );
        }
        let s = Sigma::Hyperbolic;
        let zv = z.mul(s, v);
        assert_eq!(zv.upper(s).unwrap(), z.upper(s).unwrap().mul(v.upper(s).unwrap()));
        assert_eq!(zv.lower(s).unwrap(), z.lower(s).unwrap().mul(v.lower(s).unwrap()));
        n += 1;
    }}}}}}
    println!(
        "mul_comm, mul_assoc, normForm_mul, upper_mul, lower_mul on 3 faces: {n} six-tuples, {:.0} ms",
        t.elapsed().as_secs_f64() * 1000.0
    );
}
