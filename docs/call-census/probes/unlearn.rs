//! Retained reproduction source for the `Table::unlearn` transactionality probe.
//!
//! READ-ONLY product probe. No product file is patched. This file is the
//! `src/main.rs` of a throwaway crate; it includes the **actual** production
//! sources by absolute `#[path]`:
//!
//!   * `/tmp/opencode/fold-call-census-20260911/hashtrinity/src/dense.rs`
//!   * `/tmp/opencode/fold-call-census-20260911/modelfold/src/cast/phasefold/table.rs`
//!
//! Snapshot: `/tmp/opencode/fold-call-census-20260911` @
//! `4cc14471ae24f3f204fb0b0e91736ef75c909358`.
//!
//! The only local definition is the `Monoid` trait plus its integer impls,
//! which reproduce `hashtrinity/src/algebra.rs:423-441` in shape. `algebra.rs`
//! cannot itself be `#[path]`-included because the rest of that file pulls in
//! the whole `Table` / `KeyStore` / `Coord` surface. The shim carries no
//! `Dense` or `unlearn` logic; the behaviour under test is the real file.
//!
//! Reproduce (no product build, no workspace):
//!
//! ```text
//! rustc --edition=2021 docs/call-census/probes/unlearn.rs -o /tmp/opencode/unlearn-census-probe
//! /tmp/opencode/unlearn-census-probe
//! ```
//!
//! Observed (exit 0):
//!
//! ```text
//! control Dense::try_unfuse: Err(1), table untouched  [PASS]
//! before: cnt=1 hist=[1, 0] sum=1
//! unlearn panicked: true
//! panic message: Table::unlearn: code count would go negative — nothing to remove (slot 1)
//! after:  cnt=0 hist=[1, 0] sum=1
//! cnt_mutated=true hist_unchanged=true sum_unchanged=true message_matches=true
//! FINDING REPRODUCED: the rejected unlearn left cnt half-subtracted (cnt 1->0)
//!   while hist stayed [1, 0] and sum stayed 1.0.
//! ```

extern crate self as hashtrinity;

// ---- ACTUAL production source: hashtrinity/src/dense.rs ----
#[path = "/tmp/opencode/fold-call-census-20260911/hashtrinity/src/dense.rs"]
pub mod dense;
pub use dense::{Dense, Group};

// ---- Monoid shim: algebra.rs:423-441 (trait + integer impls only) ----
pub trait Monoid {
    /// The identity element (what an absent key reads as).
    fn identity() -> Self;
    /// Combine two values for the same key (commutative and associative).
    #[must_use]
    fn combine(self, other: Self) -> Self;
}

macro_rules! int_monoid {
    ($($t:ty),*) => {$(
        impl Monoid for $t {
            #[inline] fn identity() -> Self { 0 }
            #[inline] fn combine(self, other: Self) -> Self { self + other }
        }
    )*};
}
int_monoid!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

// ---- ACTUAL production source: modelfold/.../phasefold/table.rs ----
#[path = "/tmp/opencode/fold-call-census-20260911/modelfold/src/cast/phasefold/table.rs"]
pub mod table;

use table::Table;

/// `(cnt[0], hist[0..2], sum[0])` — the three columns `unlearn` writes.
fn snapshot(t: &Table) -> (u32, Vec<u32>, f64) {
    (t.cnt.get(0), t.hist.as_slice().to_vec(), t.sum.get(0))
}

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        "<non-string panic payload>".to_string()
    }
}

fn main() {
    println!("=== Table::unlearn transactionality probe ===");
    println!("table.rs: /tmp/opencode/fold-call-census-20260911/modelfold/src/cast/phasefold/table.rs");
    println!("dense.rs: /tmp/opencode/fold-call-census-20260911/hashtrinity/src/dense.rs");
    println!("snapshot: 4cc14471ae24f3f204fb0b0e91736ef75c909358");
    println!();

    // ------------------------------------------------------------------
    // Control: one Dense column, on its own, refuses before writing.
    // ------------------------------------------------------------------
    {
        let mut d = Dense::<u32>::zeros(2);
        d.add(0, 1);
        let mut e = Dense::<u32>::zeros(2);
        e.add(1, 1);
        let before = d.clone();
        let r = d.try_unfuse(&e);
        assert_eq!(r, Err(1), "expected refusal at slot 1");
        assert_eq!(d, before, "Dense::try_unfuse must write nothing on Err");
        println!("control Dense::try_unfuse: Err(1), table untouched  [PASS]");
    }
    println!();

    // ------------------------------------------------------------------
    // Candidate: Table::unlearn checks cnt first, commits it, then hist.
    // a = one cell, k=2, code 0 once.  b = one cell, k=2, code 1 once.
    //   cnt:  a=1, b=1  -> subtract succeeds, a.cnt becomes 0.
    //   hist: a=[1,0], b=[0,1] -> slot 1 would go negative, refuses.
    // ------------------------------------------------------------------
    let mut a = Table::new(1, 2);
    a.add(0, 1.0, 0);
    let mut b = Table::new(1, 2);
    b.add(0, 1.0, 1);

    let before = snapshot(&a);
    println!("before: cnt={} hist={:?} sum={}", before.0, before.1, before.2);

    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.unlearn(&b)));
    std::panic::set_hook(prev_hook);

    let message = caught
        .as_ref()
        .err()
        .map(|p| panic_text(p.as_ref()))
        .unwrap_or_default();

    let after = snapshot(&a);
    let cnt_mutated = after.0 != before.0;
    let hist_unchanged = after.1 == before.1;
    let sum_unchanged = after.2 == before.2;
    let message_matches = message.contains("code count would go negative");

    println!("unlearn panicked: {}", caught.is_err());
    println!("panic message: {message}");
    println!("after:  cnt={} hist={:?} sum={}", after.0, after.1, after.2);
    println!(
        "cnt_mutated={cnt_mutated} hist_unchanged={hist_unchanged} \
         sum_unchanged={sum_unchanged} message_matches={message_matches}"
    );

    let reproduced = caught.is_err()
        && message_matches
        && cnt_mutated
        && hist_unchanged
        && sum_unchanged
        && after.0 == 0
        && after.1 == vec![1, 0];

    if reproduced {
        println!();
        println!(
            "FINDING REPRODUCED: the rejected unlearn left cnt half-subtracted \
             (cnt 1->0) while hist stayed [1, 0] and sum stayed 1.0."
        );
    } else {
        println!();
        println!("FINDING NOT REPRODUCED");
        std::process::exit(1);
    }
}
