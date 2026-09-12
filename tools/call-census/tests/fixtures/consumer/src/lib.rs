//! Fixture consumer. Comments deliberately contain `hashtrinity::Table` and
//! `.get()` and must not be counted. Strings likewise:
//! `let s = "hashtrinity::Table .get";`

use hashtrinity::Table; // direct import
use hashtrinity::Table as Ht; // alias import
use hashtrinity::codec::syntax::{AsTerm, Surface}; // grouped import
use hashtrinity::*; // glob import -> medium confidence
use hashtrinity::{FoldExt, SyntaxTerm}; // trait + enum
pub use hashtrinity::CoordMap; // re-export
use std::collections::HashMap;

struct Holder {
    table: Table<String, u64>,
}

trait Dyn {
    fn get(&self, k: &str) -> u64;
}

type LocalAlias = Table<String, u64>;

fn direct() {
    let s = "hashtrinity::Table .get"; // string, not a call
    let _ = s;
    let t: Table<String, u64> = Table::new();
    let _ = t.get(&"a".to_string());
    t.insert_with("b".to_string(), 1, |a, b| a + b);
    let _ = hashtrinity::crypto::blake3(b"x");
    let _ = Table::insert_with(&mut t, "c".to_string(), 2, |a, b| a + b);
    let _ = AsTerm;
    let _ = Surface;
}

fn alias() {
    let t: Ht<String, u64> = Ht::new();
    let _ = t.get(&"a".to_string());
}

fn two_on_one_line(t: &Table<String, u64>) {
    let _ = (t.get(&"a".into()), t.get(&"b".into()));
}

fn mut_ref_target(t: &mut Table<String, u64>) {
    // `& mut Table` in token form must still attribute.
    let _ = t.get(&"a".to_string());
    t.insert_with("b".to_string(), 1, |a, b| a + b);
}

fn non_target(m: &HashMap<String, u64>) {
    let _ = m.get("k"); // known non-target receiver: aggregated, not a candidate
}

fn make_opaque() -> SomeOpaque {
    todo!()
}

fn unresolved() {
    let x = make_opaque(); // return type is not inferred: receiver type unknown
    let _ = x.get(&1); // unknown receiver, `get` is a target method: candidate
    let _ = x.weird_thing(); // unknown, not a target method: only aggregated
}

fn dynamic(d: &dyn Dyn) {
    let _ = d.get("k"); // dynamic dispatch: unresolved
}

fn alias_local(a: LocalAlias) {
    let _ = a.get(&"a".into()); // local alias to a target type: attributed via alias
}

// ── Arc shadowing: the stale Table type must not survive ──────────────────
fn arc_shadow(devices: Table<String, u64>) {
    let devices = std::sync::Arc::new(devices);
    let _ = devices.clone(); // must NOT be attributed to Table::clone
}

// ── local type named Table must not be read as the target Table ───────────
mod local {
    pub struct Table {
        pub n: usize,
    }
    impl Table {
        pub fn len(&self) -> usize {
            self.n
        }
    }
}

fn local_table() {
    use crate::local::Table;
    let t = Table { n: 1 };
    let _ = t.len(); // local Table::len, not hashtrinity::Table::len
}

// ── enum and tuple-struct constructors vs associated functions ────────────
fn constructors() {
    let _ = SyntaxTerm::Bu(1); // constructor
    let _ = hashtrinity::Wrapper(2); // constructor (tuple struct)
    let _ = Table::<String, u64>::new(); // associated fn
}

// ── FoldExt provided methods on an unknown receiver ───────────────────────
fn make_iter() -> std::vec::IntoIter<u64> {
    Vec::new().into_iter()
}

fn builders() {
    let xs = make_iter();
    let _ = xs.index_by(|x| x); // FoldExt provided method, receiver unknown
    let _ = xs.count_by(|x| x);
}

#[derive(hashtrinity::Term)]
struct Row {
    a: u64,
}

macro_rules! my_macro {
    () => {
        let _ = Table::new();
    };
}

fn macro_user() {
    my_macro!();
}

// ── cfg context must be propagated ────────────────────────────────────────
#[cfg(feature = "netverify")]
fn cfg_gated(t: &Table<String, u64>) {
    let _ = t.get(&"a".to_string());
}

#[cfg(feature = "modgated")]
mod gated_mod {
    pub fn from_gated_module() {
        let _ = hashtrinity::crypto::blake3(b"x");
    }
}

#[cfg(feature = "implgated")]
impl Holder {
    pub fn from_gated_impl(&self) {
        let _ = self.table.get(&"a".to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit() {
        let t: Table<String, u64> = Table::new();
        let _ = t.get(&"a".to_string());
    }
}
