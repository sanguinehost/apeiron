//! The engine's fold faces: the fitted linears a `.fold` pack (`APFOLDP1`) replaces, and the map
//! from the pack's records onto the GGUF's linears.
//!
//! See [`crate::engine::loader`] for the loader and [`crate::engine::kern::fold`] for the product.

use crate::engine::kern::fold::{matvec_fold, FoldFace};
use crate::engine::loader::{EngineConfig, EngineError, TensorKey, TensorTable};
use crate::format::packing::FoldPlane;
/// Kinds of a coded linear, in the pack writer's order (`learn::fold::finish::train::KIND_NAMES`:
/// `q`, `k`, `v`, `o`, `gate`, `up`, `down`), which are directory parts `0..7`.
pub(super) const FOLD_KINDS: usize = 7;

/// One fitted linear: the fold faces its product is, each with the global output rows it writes, and
/// the `(o_total, n)` shape the GGUF names for it. The dense weight is never built, so the linear's
/// resident operand is the faces' own stored bytes.
#[derive(Debug)]
pub(crate) struct FoldLinear {
    /// Output rows of the linear (`o_total` in the finish tables).
    o: usize,
    /// Input columns; a multiple of the transform's 128-group.
    n: usize,
    /// `(global rows, face)`: `None` writes every row in order, `Some` a restricted plane.
    planes: Vec<(Option<Vec<u32>>, FoldFace)>,
    /// Products this linear has executed, for the route assertion that the pack's faces and not the
    /// `GGUF` payloads carry it (`Engine::fold_calls`). One per [`FoldLinear::project`] call, not one
    /// per plane, so the count is `7·n_layer` per step whatever a linear's plane split is.
    calls: std::sync::atomic::AtomicU64,
}

impl FoldLinear {
    /// The output rows written. `O(1)`.
    pub(crate) fn o(&self) -> usize {
        self.o
    }

    /// The input columns consumed. `O(1)`.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// The linear's `o` output words on the activation `x`: each plane's product placed at its
    /// global rows. `O(o·n)`; see [`matvec_fold`] for the per-call derivation.
    ///
    /// `None` on a shape disagreement or a product that leaves range.
    pub(crate) fn project(&self, x: &[i64]) -> Option<Vec<i64>> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let xs = x.get(..self.n)?;
        let mut out = vec![0_i64; self.o];
        for (rows, face) in &self.planes {
            let y = matvec_fold(face, xs)?;
            match rows {
                None => {
                    if y.len() != self.o || self.planes.len() != 1 {
                        return None;
                    }
                    out.copy_from_slice(&y);
                }
                Some(rs) => {
                    if rs.len() != y.len() {
                        return None;
                    }
                    for (v, &r) in y.iter().zip(rs.iter()) {
                        *out.get_mut(usize::try_from(r).ok()?)? = *v;
                    }
                }
            }
        }
        Some(out)
    }

    /// The faces' stored bytes: one codebook index per 8 weights, the row quantiser's `(lo, sp)`
    /// grid and 4-bit level, and the 6-/5-bit column fields. `O(planes)`.
    pub(super) fn bytes(&self) -> u64 {
        self.planes.iter().fold(0_u64, |acc, (_, f)| {
            acc.saturating_add(u64::try_from(f.bytes()).unwrap_or(u64::MAX))
        })
    }
}

/// The fitted linears of one loaded model: `linears[layer * FOLD_KINDS + kind]`, the directory index
/// of each fitted linear's GGUF entry, and the two sums [`Engine::resident_weight_bytes`] and
/// [`Engine::fold_storage`] report.
///
/// The map is derived from the pack's own write order
/// (`learn::fold::run::pack::fold_pack_of_planes`): layer-major, then `KIND_NAMES` order, then a
/// linear's planes in order, a restricted plane carrying its global `rows`. A linear's plane count is
/// not stored, so the walk takes records until their rows cover `o_total`, and a record whose
/// `(o, n)` disagrees with the config's linear refuses rather than being placed.
///
/// **Open obligation.** The record carries no manifest, so a linear whose planes *sum* two full
/// planes is indistinguishable from the next linear's first plane; such a pack is refused (the walk
/// never places a second unrestricted plane) rather than mis-assigned.
#[derive(Debug, Default)]
pub(super) struct FoldTable {
    pub(super) linears: Vec<Option<FoldLinear>>,
    /// Directory indices of the entries a fold face replaces, ascending.
    pub(super) entries: Vec<usize>,
    /// Σ of every face's stored bytes.
    pub(super) face_bytes: u64,
    /// Σ of `o·n` over the fitted linears.
    pub(super) weights: u64,
}

/// `(o_total, n)` of linear `k` of a layer, the finish tables' own geometry
/// (`learn::fold::finish::train::kind_dims`): `q` is `q_out × n_embd`, `k`/`v` are
/// `kv_out × n_embd`, `o` is `n_embd × q_out`, `gate`/`up` are `n_ff × n_embd` and `down` is
/// `n_embd × n_ff`.
///
/// # Errors
///
/// [`EngineError::ConfigMismatch`] past [`FOLD_KINDS`] or on a head-count product that leaves
/// `usize`.
pub(super) fn fold_kind_dims(c: &EngineConfig, k: usize) -> Result<(usize, usize), EngineError> {
    let q_out = c
        .n_head
        .checked_mul(c.head_dim)
        .ok_or_else(|| EngineError::ConfigMismatch(String::from("fold pack: q_out")))?;
    let kv_out = c
        .n_head_kv
        .checked_mul(c.head_dim)
        .ok_or_else(|| EngineError::ConfigMismatch(String::from("fold pack: kv_out")))?;
    match k {
        0 => Ok((q_out, c.n_embd)),
        1 | 2 => Ok((kv_out, c.n_embd)),
        3 => Ok((c.n_embd, q_out)),
        4 | 5 => Ok((c.n_ff, c.n_embd)),
        6 => Ok((c.n_embd, c.n_ff)),
        _ => Err(EngineError::ConfigMismatch(format!(
            "fold pack: linear kind {k} >= {FOLD_KINDS}"
        ))),
    }
}

/// The fitted linears of `config` from the pack's records, in the writer's order, and the directory
/// entry each replaces. `O(planes + 7·n_layer·entries)`; a restricted plane contributes its faces in
/// record order and its global rows from its own `rows` field.
///
/// # Errors
///
/// [`EngineError::ConfigMismatch`] on a plane count the config's linears do not account for, a
/// record whose `(o, n)` is not the linear's, a record no [`FoldFace`](super::kern::fold::FoldFace)
/// carries, or a linear whose records' rows do not cover it exactly.
pub(super) fn build_fold_table(
    config: &EngineConfig,
    table: &TensorTable,
    planes: &[FoldPlane],
) -> Result<FoldTable, EngineError> {
    let mut out = FoldTable {
        linears: (0..config.n_layer.saturating_mul(FOLD_KINDS))
            .map(|_| None)
            .collect(),
        ..FoldTable::default()
    };
    let entries = table.entries();
    let mut ri = 0usize;
    for layer in 0..config.n_layer {
        for k in 0..FOLD_KINDS {
            let (o_total, n) = fold_kind_dims(config, k)?;
            let mut ps: Vec<(Option<Vec<u32>>, FoldFace)> = Vec::new();
            let mut covered = 0usize;
            while covered < o_total {
                let Some(rec) = planes.get(ri) else {
                    return Err(EngineError::ConfigMismatch(format!(
                        "fold pack: {} planes end at blk.{layer} kind {k}, {}/{} rows",
                        planes.len(),
                        covered,
                        o_total
                    )));
                };
                ri += 1;
                if rec.n != n || rec.o == 0 {
                    return Err(EngineError::ConfigMismatch(format!(
                        "fold pack: plane {ri} is {}x{}, blk.{layer} kind {k} is {o_total}x{n}",
                        rec.o, rec.n
                    )));
                }
                let face = FoldFace::from_plane(rec).ok_or_else(|| {
                    EngineError::ConfigMismatch(format!(
                        "fold pack: plane {ri} is not a codebook face (blk.{layer} kind {k})"
                    ))
                })?;
                match &rec.rows {
                    None => {
                        if !ps.is_empty() {
                            return Err(EngineError::ConfigMismatch(format!(
                                "fold pack: plane {ri} is a second unrestricted plane of \
                                 blk.{layer} kind {k}, which the record cannot mark"
                            )));
                        }
                        covered = covered.saturating_add(rec.o);
                        ps.push((None, face));
                    }
                    Some(rows) => {
                        if rows.len() != rec.o {
                            return Err(EngineError::ConfigMismatch(format!(
                                "fold pack: plane {ri} carries {} rows for {} output rows",
                                rows.len(),
                                rec.o
                            )));
                        }
                        covered = covered.saturating_add(rows.len());
                        ps.push((Some(rows.clone()), face));
                    }
                }
            }
            if covered != o_total {
                return Err(EngineError::ConfigMismatch(format!(
                    "fold pack: blk.{layer} kind {k} rows cover {covered} of {o_total}"
                )));
            }
            let linear = FoldLinear {
                o: o_total,
                n,
                planes: ps,
                calls: std::sync::atomic::AtomicU64::new(0),
            };
            out.face_bytes = out.face_bytes.saturating_add(linear.bytes());
            out.weights = out
                .weights
                .saturating_add(u64::try_from(o_total.saturating_mul(n)).unwrap_or(u64::MAX));
            let mut ei = 0usize;
            while ei < entries.len() {
                if crate::engine::step::key_matches(
                    entries[ei].name.as_bytes(),
                    TensorKey::blk(layer, u8::try_from(k).unwrap_or(u8::MAX)),
                ) {
                    out.entries.push(ei);
                }
                ei += 1;
            }
            out.linears[layer * FOLD_KINDS + k] = Some(linear);
        }
    }
    if ri != planes.len() {
        return Err(EngineError::ConfigMismatch(format!(
            "fold pack: {} planes for {ri} of {} fitted linears",
            planes.len(),
            config.n_layer.saturating_mul(FOLD_KINDS)
        )));
    }
    out.entries.sort_unstable();
    out.entries.dedup();
    Ok(out)
}

impl super::Engine {
    /// The fold products this engine has executed: one per [`FoldLinear::project`] call, summed over
    /// the fitted linears. `O(n_layer·FOLD_KINDS)`.
    #[must_use]
    pub fn fold_calls(&self) -> u64 {
        self.fold.linears.iter().flatten().fold(0_u64, |acc, l| {
            acc.saturating_add(l.calls.load(std::sync::atomic::Ordering::Relaxed))
        })
    }
}

#[cfg(test)]
mod test {
    use super::super::{Engine, EngineError};

    /// `Engine::open_fold` reads the pack before the GGUF, so a missing pair refuses through the
    /// pack reader rather than opening either file. Roots the entry point on the crate's own path.
    #[test]
    fn open_fold_refuses_a_missing_pack() {
        let e = Engine::open_fold("/nonexistent/model.gguf", "/nonexistent/model.fold");
        assert!(matches!(e, Err(EngineError::ConfigMismatch(_))));
    }
}
