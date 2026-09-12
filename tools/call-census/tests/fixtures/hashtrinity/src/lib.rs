//! Minimal stand-in for the target crate. Only the shapes the census needs to
//! recognise: faces, methods, trait-provided builders, enum and tuple-struct
//! constructors, a free function, a module, and an exported derive-name macro.

pub struct Table<K, V> {
    _k: core::marker::PhantomData<K>,
    _v: core::marker::PhantomData<V>,
}

impl<K, V> Table<K, V> {
    pub fn new() -> Self {
        Table { _k: core::marker::PhantomData, _v: core::marker::PhantomData }
    }
    pub fn get(&self, _k: &K) -> Option<&V> {
        None
    }
    pub fn len(&self) -> usize {
        0
    }
    pub fn insert_with(&mut self, _k: K, _v: V, _f: impl FnOnce(V, V) -> V) {}
}

pub struct CoordMap<K, V> {
    _k: core::marker::PhantomData<K>,
    _v: core::marker::PhantomData<V>,
}

impl<K, V> CoordMap<K, V> {
    pub fn get(&self, _k: &K) -> Option<&V> {
        None
    }
}

pub struct FreqMap<K, V> {
    _k: core::marker::PhantomData<K>,
    _v: core::marker::PhantomData<V>,
}

impl<K, V> FreqMap<K, V> {
    pub fn get(&self, _k: &K) -> Option<&V> {
        None
    }
    pub fn count(&self, _k: &K) -> i64 {
        0
    }
}

/// A trait with *provided* methods. These must be in the inventory even though
/// no `impl` block mentions them.
pub trait FoldExt: Iterator + Sized {
    fn index_by<K, F>(self, _f: F) -> Table<K, usize>
    where
        F: FnMut(Self::Item) -> K,
    {
        let _ = _f;
        Table::new()
    }
    fn set_by<K, F>(self, _f: F) -> Table<K, ()>
    where
        F: FnMut(Self::Item) -> K,
    {
        let _ = _f;
        Table::new()
    }
    fn count_by<K, F>(self, _f: F) -> Table<K, i64>
    where
        F: FnMut(Self::Item) -> K,
    {
        let _ = _f;
        Table::new()
    }
}

impl<I: Iterator> FoldExt for I {}

pub enum SyntaxTerm {
    Bu(u64),
    Str(String),
}

pub struct Wrapper(pub u64);

pub mod crypto {
    pub fn blake3(_bytes: &[u8]) -> [u8; 32] {
        [0; 32]
    }
}

pub mod codec {
    pub mod syntax {
        pub struct AsTerm;
        pub struct Surface;
    }
}

#[macro_export]
macro_rules! Term {
    () => {};
}

pub use crypto::blake3;

#[cfg(test)]
mod tests {
    /// Must NOT enter the target inventory.
    pub struct Engine {
        pub x: u64,
    }
    impl Engine {
        pub fn test_only_method(&self) -> u64 {
            self.x
        }
    }
    #[test]
    fn t() {}
}
