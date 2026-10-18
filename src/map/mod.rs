//! Every `K -> V` map in the crate and the key addressing it rests on: the growable flat/spill
//! [`table`], the subsumed incumbent `hashtrinity` (public face [`freq`]), the coordinate-keyed
//! counting container [`count`], key addressing ([`coord`], [`fold`], [`pepper`]), the storage
//! faces ([`keystore`]) and the [`merge`]s.

pub mod coord;
pub mod count;
pub mod fold;
/// The incumbent's exact NB count face (`FreqMap`) and map face (`Table`), ported into this crate.
pub mod freq;
// `hashtrinity`, ported from the author's `fold` repository at rev `4cc14471` with its upstream
// layout. The host lint policy is scoped off it by the module-level `#![allow]` in
// `src/map/hashtrinity/mod.rs`; `rustfmt` is scoped off it by the `#[rustfmt::skip]` below.
/// The subsumed incumbent: `hashtrinity` at rev `4cc14471`, ported as this crate's own module.
/// Private, because `freq` is its public face. The external `hashtrinity` git dependency is
/// removed, so the doctests in the submodules are marked `ignore`:
/// they document the upstream crate's API, and this crate's public path for the port is `freq`.
#[rustfmt::skip]
mod hashtrinity;
pub mod keystore;
pub mod merge;
pub mod pepper;
pub mod table;
