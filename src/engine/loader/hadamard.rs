//! The `prism.hadamard.*` metadata contract: parse and validate the fork's Hadamard fold keys.

use crate::format::gguf::{GVal, GgufHeader};

use super::EngineError;

/// The `normalized-sylvester-walsh-hadamard` transform name the fork's runtime accepts.
pub(super) const HADAMARD_TRANSFORM: &str = "normalized-sylvester-walsh-hadamard";
/// The `input-last-dimension` axis name the fork's runtime accepts.
pub(super) const HADAMARD_AXIS: &str = "input-last-dimension";

/// The `PrismML` `prism.hadamard.*` fold contract (`llama-model.cpp:1197-1263`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HadamardMeta {
    /// `prism.hadamard.version`: `1` for an untied output, `2` for a tied output.
    pub version: u32,
    /// `prism.hadamard.block_size`: the rotation block width, a nonzero power of two.
    pub block_size: u32,
    /// `prism.hadamard.tied_output`: true iff the output head reuses the rotated embedding.
    pub tied_output: bool,
    /// `prism.hadamard.weight_names`: folded tensor names, never empty.
    pub weight_names: Vec<String>,
    /// `prism.hadamard.sign_widths` paired with `sign_values`; `None` for `sign_mode = "identity"`.
    pub signs: Option<Vec<(u32, Vec<i8>)>>,
}

/// `h.meta(key)` as an `i64`; `Ok(None)` when absent, a refusal when present but not an `INT` type
/// in `i64` range.
fn hadamard_i64(h: &GgufHeader, key: &str) -> Result<Option<i64>, EngineError> {
    match h.meta(key) {
        None => Ok(None),
        Some(GVal::Int(x, _)) => match i64::try_from(*x) {
            Ok(v) => Ok(Some(v)),
            Err(_) => Err(EngineError::BadMetadata(format!("{key} is not an integer"))),
        },
        Some(_) => Err(EngineError::BadMetadata(format!("{key} is not an integer"))),
    }
}

fn hadamard_require_i64(h: &GgufHeader, key: &str) -> Result<i64, EngineError> {
    hadamard_i64(h, key)?.ok_or_else(|| EngineError::MissingMetadata(key.to_string()))
}

fn hadamard_require_string(h: &GgufHeader, key: &str) -> Result<String, EngineError> {
    match h.meta(key) {
        None => Err(EngineError::MissingMetadata(key.to_string())),
        Some(GVal::Str(s)) => Ok(String::from_utf8_lossy(s).into_owned()),
        Some(_) => Err(EngineError::BadMetadata(format!("{key} is not a string"))),
    }
}

/// A string-array key; `Ok(None)` when absent. Cost is O(elements) time and space.
fn hadamard_string_array(h: &GgufHeader, key: &str) -> Result<Option<Vec<String>>, EngineError> {
    match h.meta(key) {
        None => Ok(None),
        Some(GVal::Arr(_, elems)) => {
            let mut out: Vec<String> = Vec::with_capacity(elems.len());
            let mut i = 0usize;
            let mut err: Option<EngineError> = None;
            while i < elems.len() && err.is_none() {
                match &elems[i] {
                    GVal::Str(s) => out.push(String::from_utf8_lossy(s).into_owned()),
                    _ => {
                        err = Some(EngineError::BadMetadata(format!(
                            "{key}[{i}] is not a string"
                        )));
                    }
                }
                i += 1;
            }
            if let Some(e) = err {
                return Err(e);
            }
            Ok(Some(out))
        }
        Some(_) => Err(EngineError::BadMetadata(format!("{key} is not an array"))),
    }
}

fn hadamard_int_array(h: &GgufHeader, key: &str) -> Result<Option<Vec<i64>>, EngineError> {
    match h.meta(key) {
        None => Ok(None),
        Some(GVal::Arr(_, elems)) => {
            let mut out: Vec<i64> = Vec::with_capacity(elems.len());
            let mut i = 0usize;
            let mut err: Option<EngineError> = None;
            while i < elems.len() && err.is_none() {
                match elems[i].as_i64() {
                    Some(v) => out.push(v),
                    None => {
                        err = Some(EngineError::BadMetadata(format!(
                            "{key}[{i}] is not an integer"
                        )));
                    }
                }
                i += 1;
            }
            if let Some(e) = err {
                return Err(e);
            }
            Ok(Some(out))
        }
        Some(_) => Err(EngineError::BadMetadata(format!("{key} is not an array"))),
    }
}

/// Validate the `explicit` sign table into `(width, values)` pairs; cost is O(sum of widths).
fn hadamard_signs(h: &GgufHeader, block_size: u32) -> Result<Vec<(u32, Vec<i8>)>, EngineError> {
    let widths = hadamard_int_array(h, "prism.hadamard.sign_widths")?.unwrap_or_default();
    let values = hadamard_int_array(h, "prism.hadamard.sign_values")?.unwrap_or_default();
    if widths.is_empty() {
        return Err(EngineError::BadMetadata(
            "prism.hadamard.sign_mode is explicit but sign_widths is empty".to_string(),
        ));
    }
    let mut out: Vec<(u32, Vec<i8>)> = Vec::with_capacity(widths.len());
    let mut off = 0usize;
    let mut i = 0usize;
    let mut err: Option<EngineError> = None;
    while i < widths.len() && err.is_none() {
        let width = widths[i];
        let width_u32 = u32::try_from(width)
            .ok()
            .filter(|w| *w != 0 && w % block_size == 0);
        match width_u32 {
            None => {
                err = Some(EngineError::BadMetadata(format!(
                    "invalid prism.hadamard sign width {width}: not a positive multiple of block size {block_size}"
                )));
            }
            Some(width_u32) => {
                let width_usize = width_u32 as usize;
                if off + width_usize > values.len() {
                    err = Some(EngineError::BadMetadata(format!(
                        "invalid prism.hadamard sign width {width}: sign_values is too short"
                    )));
                } else {
                    let mut vec: Vec<i8> = Vec::with_capacity(width_usize);
                    let mut j = off;
                    while j < off + width_usize && err.is_none() {
                        let v = values[j];
                        if v != 1 && v != -1 {
                            err = Some(EngineError::BadMetadata(
                                "prism.hadamard sign values must be +/-1".to_string(),
                            ));
                        } else {
                            vec.push(if v == 1 { 1i8 } else { -1i8 });
                        }
                        j += 1;
                    }
                    if err.is_none() {
                        out.push((width_u32, vec));
                        off += width_usize;
                    }
                }
            }
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    if off != values.len() {
        return Err(EngineError::BadMetadata(
            "prism.hadamard.sign_values length mismatch".to_string(),
        ));
    }
    Ok(out)
}

/// Read and validate the `prism.hadamard.*` metadata: an absent version key is `Ok(None)`, and
/// otherwise every value the fork's loader rejects is a refusal. Cost is O(metadata size).
///
/// # Errors
///
/// [`EngineError::MissingMetadata`] or [`EngineError::BadMetadata`] when the fork would refuse.
pub fn hadamard_meta(h: &GgufHeader) -> Result<Option<HadamardMeta>, EngineError> {
    let Some(version) = hadamard_i64(h, "prism.hadamard.version")? else {
        return Ok(None);
    };
    let version = u32::try_from(version).map_err(|_| {
        EngineError::BadMetadata(format!("prism.hadamard.version {version} is not 1 or 2"))
    })?;
    if version != 1 && version != 2 {
        return Err(EngineError::BadMetadata(format!(
            "prism.hadamard.version {version} is not 1 or 2"
        )));
    }

    let tied_output = match h.meta("prism.hadamard.tied_output") {
        None => false,
        Some(GVal::Bool(b)) => *b,
        Some(_) => {
            return Err(EngineError::BadMetadata(
                "prism.hadamard.tied_output is not a bool".to_string(),
            ))
        }
    };
    if (version == 2) != tied_output {
        return Err(EngineError::BadMetadata(
            "prism.hadamard version 2 requires tied_output=true; version 1 forbids it".to_string(),
        ));
    }

    let block_size = hadamard_require_i64(h, "prism.hadamard.block_size")?;
    let block_size = u32::try_from(block_size).map_err(|_| {
        EngineError::BadMetadata("prism.hadamard.block_size is not a positive integer".to_string())
    })?;
    if block_size == 0 || (block_size & (block_size - 1)) != 0 {
        return Err(EngineError::BadMetadata(format!(
            "prism.hadamard.block_size {block_size} is zero or not a power of two"
        )));
    }

    let transform = hadamard_require_string(h, "prism.hadamard.transform")?;
    if transform != HADAMARD_TRANSFORM {
        return Err(EngineError::BadMetadata(format!(
            "unsupported prism.hadamard.transform {transform:?}"
        )));
    }
    let axis = hadamard_require_string(h, "prism.hadamard.axis")?;
    if axis != HADAMARD_AXIS {
        return Err(EngineError::BadMetadata(format!(
            "unsupported prism.hadamard.axis {axis:?}"
        )));
    }
    let sign_mode = hadamard_require_string(h, "prism.hadamard.sign_mode")?;
    if sign_mode != "identity" && sign_mode != "explicit" {
        return Err(EngineError::BadMetadata(format!(
            "unsupported prism.hadamard.sign_mode {sign_mode:?}"
        )));
    }

    let weight_names = hadamard_string_array(h, "prism.hadamard.weight_names")?
        .ok_or_else(|| EngineError::MissingMetadata("prism.hadamard.weight_names".to_string()))?;
    if weight_names.is_empty() {
        return Err(EngineError::BadMetadata(
            "prism.hadamard.weight_names is empty".to_string(),
        ));
    }

    let signs = if sign_mode == "explicit" {
        Some(hadamard_signs(h, block_size)?)
    } else {
        None
    };

    Ok(Some(HadamardMeta {
        version,
        block_size,
        tied_output,
        weight_names,
        signs,
    }))
}
