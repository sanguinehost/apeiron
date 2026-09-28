//! The pure SPIR-V DP4A fragment checker.
//!
//! [`check_fragment`] decodes the binary shape of an emitted SPIR-V 1.3 module and applies the
//! fragment obligations mirrored from the proved `Spirv.Fragment`/`Spirv.Decode` Lean modules:
//! the five-word header, `WordCount` framing (D1), `result-id` uniqueness and
//! definition-before-use (D2), the `OpSUDot` integer type agreement (D3), admitted storage
//! classes and decorations (D4), and the served opcode set of `Spirv.Fragment.IsDp4aFragment`.
//!
//! The checker performs no float semantics: the float epilogue opcodes (`OpExtInst`,
//! `OpTypeFloat`, `OpConvertSToF`, `OpFAdd`, `OpFMul`) are admitted syntax and are validated only
//! for their binary shape. It never panics on input and never unwraps parsed data.
//!
//! This body is the extraction root for `Spirv.Check` in wave 3.
//!
//! One deliberate divergence from the current `Spirv.Decode` scan is recorded here and is a
//! residual for that wave:
//!
//! * Annotation and execution-mode operands (`OpDecorate` target, `OpMemberDecorate` type,
//!   `OpExecutionMode` entry) are the SPIR-V forward references the emitter emits; SPIR-V 1.3 §2.4
//!   allows them, while `Inst.uses` currently lists them as definition-before-use uses and so
//!   refuses the served module. The checker exempts exactly those positions.
//!
//! `SERVED_OPCODES` is the mechanically derived 42-opcode served set. `OpShiftLeftLogical` (196)
//! and `OpBitwiseOr` (197) are off the served path (`moe`/`two_level`/`q3` only), are absent from
//! `Decode.decodeInst` and are refused as not admitted.
//!
//! The scan state (defined ids, id-to-tag table) is held in insertion-ordered `Vec`s searched by
//! index rather than `std::collections::{HashSet,HashMap}`. Aeneas models neither hash collection:
//! extracting this body with them leaves an unmodeled `HashMap`/`HashSet`/hasher as axioms, which
//! `proof/regen.sh`'s fail-closed audit rejects. The vector forms accept and refuse exactly the
//! same word streams; only the lookup cost changes (linear, stated at [`check_fragment`]).

/// Words in the SPIR-V header: `magic`, version, generator, bound, schema.
const HEADER_WORDS: usize = 5;

/// The SPIR-V magic number.
const MAGIC: u32 = 0x0723_0203;

/// The SPIR-V 1.3 version word.
const VERSION_1_3: u32 = 0x0001_0300;

/// The opcode set the emitter's served path can emit, from `Spirv.Fragment.servedOpcodes` (42).
const SERVED_OPCODES: [u32; 42] = [
    10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 29, 30, 32, 33, 43, 54, 56, 59, 61, 62, 65, 71,
    72, 81, 111, 128, 129, 132, 133, 169, 176, 194, 199, 246, 247, 248, 249, 250, 253, 4452,
];

/// Storage classes the fragment admits: `Input`, `Workgroup`, `Function`, `StorageBuffer`.
const VALID_STORAGE_CLASSES: [u32; 4] = [1, 4, 7, 12];

/// Decorations the fragment admits.
const VALID_DECORATIONS: [u32; 6] = [2, 6, 11, 33, 34, 35];

/// A typed refusal from [`check_fragment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckError {
    /// The word list is shorter than the five-word SPIR-V header.
    HeaderTooShort {
        /// Number of words supplied.
        words: usize,
    },
    /// The first header word is not the SPIR-V magic number.
    BadMagic {
        /// The word found in its place.
        found: u32,
    },
    /// The version word is not SPIR-V 1.3.
    BadVersion {
        /// The word found in its place.
        found: u32,
    },
    /// The schema word is not zero.
    BadSchema {
        /// The word found in its place.
        found: u32,
    },
    /// The id bound is zero.
    ZeroBound,
    /// An instruction word carried a zero `WordCount`, leaving its boundary undefined (D1).
    ZeroWordCount {
        /// Word index of the instruction header.
        index: usize,
    },
    /// An instruction's `WordCount` overruns the remaining words (D1).
    TruncatedInstruction {
        /// Word index of the instruction header.
        index: usize,
        /// The claimed `WordCount`.
        word_count: u32,
    },
    /// An instruction's opcode is outside the admitted served set.
    OpcodeNotAdmitted {
        /// Word index of the instruction header.
        index: usize,
        /// The opcode found.
        opcode: u32,
    },
    /// An instruction's operand count does not match its opcode's binary form.
    MalformedInstruction {
        /// Word index of the instruction header.
        index: usize,
        /// The opcode found.
        opcode: u32,
    },
    /// An `OpSUDot` packed-format operand is not `PackedVectorFormat4x8Bit` (`0`).
    BadPackedFormat {
        /// Word index of the instruction header.
        index: usize,
        /// The format operand found.
        format: u32,
    },
    /// A used id is not defined by the instruction prefix (D2).
    UndefinedId {
        /// Word index of the instruction header.
        index: usize,
        /// The undefined id.
        id: u32,
    },
    /// A result id was already defined (D2).
    DuplicateResultId {
        /// Word index of the instruction header.
        index: usize,
        /// The repeated id.
        id: u32,
    },
    /// An `OpSUDot` result type or operand is not a 32-bit integer type (D3).
    TypeMismatch {
        /// Word index of the instruction header.
        index: usize,
    },
    /// A storage class is outside the admitted set (D4).
    BadStorageClass {
        /// Word index of the instruction header.
        index: usize,
        /// The storage class found.
        storage_class: u32,
    },
    /// A decoration is outside the admitted set (D4).
    BadDecoration {
        /// Word index of the instruction header.
        index: usize,
        /// The decoration found.
        decoration: u32,
    },
    /// An `OpEntryPoint` literal string has no zero-byte terminator.
    UnterminatedLiteral {
        /// Word index of the instruction header.
        index: usize,
    },
}

/// Check a SPIR-V word stream against the DP4A fragment obligations.
///
/// The stream must begin with the five-word header (`magic`, version, generator, bound, schema)
/// followed by complete instructions. `magic` must be the SPIR-V magic number, the version must be
/// SPIR-V 1.3, the schema word must be zero and the id bound must be nonzero. Every instruction
/// must frame with a nonzero `WordCount` that stays inside the stream, carry an opcode in the
/// admitted served set with a matching operand count, define unique result ids after their data
/// operands, and satisfy the `OpSUDot` type, storage-class and decoration obligations.
///
/// # Errors
///
/// Returns the specific [`CheckError`] for the first obligation the stream violates; the variant
/// carries the word index of the offending instruction where one applies.
///
/// Temporal cost is O(W²) word comparisons in the word count W: the scan performs one linear pass
/// over the words, and each instruction's id/tag obligation costs a linear search of the defined-id
/// and id-to-tag vectors (the derivations are in [`defs_contains`] and [`ty_get`]). Space is O(W).
/// The emitted `gemv_dp4a_module_dyn` streams are a few hundred words, so the quadratic term is
/// bounded and is the price of an Aeneas-representable scan; the accepted language is unchanged.
pub fn check_fragment(words: &[u32]) -> Result<(), CheckError> {
    if words.len() < HEADER_WORDS {
        return Err(CheckError::HeaderTooShort { words: words.len() });
    }
    if words[0] != MAGIC {
        return Err(CheckError::BadMagic { found: words[0] });
    }
    if words[1] != VERSION_1_3 {
        return Err(CheckError::BadVersion { found: words[1] });
    }
    if words[3] == 0 {
        return Err(CheckError::ZeroBound);
    }
    if words[4] != 0 {
        return Err(CheckError::BadSchema { found: words[4] });
    }

    let mut defs: Vec<u32> = Vec::new();
    let mut tys: Vec<(u32, TyTag)> = Vec::new();
    let mut index = HEADER_WORDS;
    while index < words.len() {
        let first = words[index];
        let word_count = first >> 16;
        let opcode = first & 0xffff;
        if word_count == 0 {
            return Err(CheckError::ZeroWordCount { index });
        }
        let end = index + word_count as usize;
        if end > words.len() {
            return Err(CheckError::TruncatedInstruction { index, word_count });
        }
        let operands = &words[index + 1..end];
        let shape = shape_of(index, opcode, operands)?;
        apply_shape(index, &shape, &mut defs, &mut tys)?;
        index = end;
    }
    Ok(())
}

/// A shallow type tag, mirroring `Spirv.Decode.TyTag`. Widths are kept for the `OpSUDot` check;
/// signedness is not needed by any fragment obligation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TyTag {
    /// `OpTypeInt` of the given width.
    Int(u32),
    /// `OpTypeBool`.
    Bool,
    /// `OpTypeFloat` of the given width.
    Float(u32),
    /// `OpTypeVector`.
    Vec,
    /// `OpTypePointer`.
    Pointer,
    /// `OpTypeRuntimeArray`.
    RuntimeArray,
    /// `OpTypeFunction`.
    Function,
    /// `OpTypeStruct`.
    Struct,
    /// `OpTypeVoid`.
    Void,
}

/// How the result tag of an instruction is resolved, mirroring `Spirv.Decode.resultTag`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TagSpec {
    /// The instruction defines no typed value.
    None,
    /// `OpTypeVoid`.
    Void,
    /// `OpTypeBool`.
    Bool,
    /// `OpTypeInt` of this width.
    Int(u32),
    /// `OpTypeFloat` of this width.
    Float(u32),
    /// `OpTypeVector`; the element tag is looked up from this id.
    Vector(u32),
    /// `OpTypeRuntimeArray`; the element tag is looked up from this id.
    RuntimeArray(u32),
    /// `OpTypePointer`; the pointee tag is looked up from this id.
    Pointer(u32),
    /// `OpTypeFunction`.
    Function,
    /// `OpTypeStruct`.
    Struct,
    /// The result adopts the tag of this type id.
    OfType(u32),
}

/// The operand fields one instruction contributes to the scan.
struct Shape {
    /// The result id the instruction defines, if any.
    result: Option<u32>,
    /// The ids the instruction's operands consume.
    uses: Vec<u32>,
    /// How the result tag resolves.
    tag: TagSpec,
    /// The storage class operand, when present.
    storage_class: Option<u32>,
    /// The decoration operand, when present.
    decoration: Option<u32>,
    /// The `OpSUDot` `(result type, operand a, operand b)` triple, when present.
    sudot: Option<(u32, u32, u32)>,
}

/// A shape with no result, operands, tags or obligations.
const fn plain() -> Shape {
    Shape {
        result: None,
        uses: Vec::new(),
        tag: TagSpec::None,
        storage_class: None,
        decoration: None,
        sudot: None,
    }
}

/// A one-element operand vector. The `vec![..]` macro's array-to-`Vec` lowering drags an unmodeled
/// `core::mem::maybe_uninit::MaybeUninit` axiom through Aeneas, so the operand vectors are built
/// with explicit pushes instead.
#[expect(
    clippy::vec_init_then_push,
    reason = "the vec![] macro drags an unmodeled core::mem::maybe_uninit::MaybeUninit axiom \
              through the Aeneas extraction; the explicit pushes are what keep the extraction \
              axiom-free"
)]
fn ids1(a: u32) -> Vec<u32> {
    let mut v = Vec::new();
    v.push(a);
    v
}

/// A two-element operand vector, built without `vec![..]` for the reason at [`ids1`].
fn ids2(a: u32, b: u32) -> Vec<u32> {
    let mut v = ids1(a);
    v.push(b);
    v
}

/// A three-element operand vector, built without `vec![..]` for the reason at [`ids1`].
fn ids3(a: u32, b: u32, c: u32) -> Vec<u32> {
    let mut v = ids2(a, b);
    v.push(c);
    v
}

/// A four-element operand vector, built without `vec![..]` for the reason at [`ids1`].
fn ids4(w0: u32, w1: u32, w2: u32, w3: u32) -> Vec<u32> {
    let mut v = ids3(w0, w1, w2);
    v.push(w3);
    v
}

/// Refuse an instruction whose operand count is not exactly `expected`.
const fn require_exact(
    index: usize,
    opcode: u32,
    operands: &[u32],
    expected: usize,
) -> Result<(), CheckError> {
    if operands.len() == expected {
        Ok(())
    } else {
        Err(CheckError::MalformedInstruction { index, opcode })
    }
}

/// Refuse an instruction whose operand count is below `minimum`.
const fn require_min(
    index: usize,
    opcode: u32,
    operands: &[u32],
    minimum: usize,
) -> Result<(), CheckError> {
    if operands.len() >= minimum {
        Ok(())
    } else {
        Err(CheckError::MalformedInstruction { index, opcode })
    }
}

/// Decode one instruction into its scan shape, refusing an unadmitted opcode or a wrong operand
/// count. Mirrors `Spirv.Decode.decodeInst`.
#[expect(
    clippy::too_many_lines,
    reason = "shape_of is the one-to-one match over the served opcode binary forms; splitting it \
              would scatter the operand-count table that mirrors decodeInst"
)]
fn shape_of(index: usize, opcode: u32, operands: &[u32]) -> Result<Shape, CheckError> {
    if !served_contains(opcode) {
        return Err(CheckError::OpcodeNotAdmitted { index, opcode });
    }
    let shape = match opcode {
        17 | 249 => {
            require_exact(index, opcode, operands, 1)?;
            plain()
        }
        10 => plain(),
        11 => {
            require_min(index, opcode, operands, 1)?;
            Shape {
                result: Some(operands[0]),
                ..plain()
            }
        }
        12 => {
            require_min(index, opcode, operands, 4)?;
            let mut uses = ids2(operands[0], operands[2]);
            uses.extend_from_slice(&operands[4..]);
            Shape {
                result: Some(operands[1]),
                uses,
                tag: TagSpec::OfType(operands[0]),
                ..plain()
            }
        }
        14 | 247 => {
            require_exact(index, opcode, operands, 2)?;
            plain()
        }
        15 => {
            require_min(index, opcode, operands, 2)?;
            if !has_literal_terminator(&operands[2..]) {
                return Err(CheckError::UnterminatedLiteral { index });
            }
            plain()
        }
        16 => {
            require_min(index, opcode, operands, 2)?;
            plain()
        }
        71 => {
            require_min(index, opcode, operands, 2)?;
            Shape {
                decoration: Some(operands[1]),
                ..plain()
            }
        }
        72 => {
            require_min(index, opcode, operands, 3)?;
            Shape {
                decoration: Some(operands[2]),
                ..plain()
            }
        }
        19 => {
            require_exact(index, opcode, operands, 1)?;
            Shape {
                result: Some(operands[0]),
                tag: TagSpec::Void,
                ..plain()
            }
        }
        20 => {
            require_exact(index, opcode, operands, 1)?;
            Shape {
                result: Some(operands[0]),
                tag: TagSpec::Bool,
                ..plain()
            }
        }
        21 => {
            require_exact(index, opcode, operands, 3)?;
            Shape {
                result: Some(operands[0]),
                tag: TagSpec::Int(operands[1]),
                ..plain()
            }
        }
        22 => {
            require_exact(index, opcode, operands, 2)?;
            Shape {
                result: Some(operands[0]),
                tag: TagSpec::Float(operands[1]),
                ..plain()
            }
        }
        23 => {
            require_exact(index, opcode, operands, 3)?;
            Shape {
                result: Some(operands[0]),
                uses: ids1(operands[1]),
                tag: TagSpec::Vector(operands[1]),
                ..plain()
            }
        }
        29 => {
            require_exact(index, opcode, operands, 2)?;
            Shape {
                result: Some(operands[0]),
                uses: ids1(operands[1]),
                tag: TagSpec::RuntimeArray(operands[1]),
                ..plain()
            }
        }
        30 => {
            require_min(index, opcode, operands, 1)?;
            Shape {
                result: Some(operands[0]),
                uses: operands[1..].to_vec(),
                tag: TagSpec::Struct,
                ..plain()
            }
        }
        32 => {
            require_exact(index, opcode, operands, 3)?;
            Shape {
                result: Some(operands[0]),
                uses: ids1(operands[2]),
                tag: TagSpec::Pointer(operands[2]),
                storage_class: Some(operands[1]),
                ..plain()
            }
        }
        33 => {
            require_min(index, opcode, operands, 2)?;
            Shape {
                result: Some(operands[0]),
                uses: operands[1..].to_vec(),
                tag: TagSpec::Function,
                ..plain()
            }
        }
        43 => {
            require_exact(index, opcode, operands, 3)?;
            Shape {
                result: Some(operands[1]),
                uses: ids1(operands[0]),
                tag: TagSpec::OfType(operands[0]),
                ..plain()
            }
        }
        54 => {
            require_exact(index, opcode, operands, 4)?;
            Shape {
                result: Some(operands[1]),
                uses: ids2(operands[0], operands[3]),
                ..plain()
            }
        }
        56 | 253 => {
            require_exact(index, opcode, operands, 0)?;
            plain()
        }
        59 => match operands.len() {
            3 => Shape {
                result: Some(operands[1]),
                uses: ids1(operands[0]),
                tag: TagSpec::OfType(operands[0]),
                storage_class: Some(operands[2]),
                ..plain()
            },
            4 => Shape {
                result: Some(operands[1]),
                uses: ids2(operands[0], operands[3]),
                tag: TagSpec::OfType(operands[0]),
                storage_class: Some(operands[2]),
                ..plain()
            },
            _ => return Err(CheckError::MalformedInstruction { index, opcode }),
        },
        61 | 111 => {
            require_exact(index, opcode, operands, 3)?;
            Shape {
                result: Some(operands[1]),
                uses: ids2(operands[0], operands[2]),
                tag: TagSpec::OfType(operands[0]),
                ..plain()
            }
        }
        62 => {
            require_exact(index, opcode, operands, 2)?;
            Shape {
                uses: ids2(operands[0], operands[1]),
                ..plain()
            }
        }
        65 => {
            require_min(index, opcode, operands, 3)?;
            let mut uses = ids2(operands[0], operands[2]);
            uses.extend_from_slice(&operands[3..]);
            Shape {
                result: Some(operands[1]),
                uses,
                tag: TagSpec::OfType(operands[0]),
                ..plain()
            }
        }
        81 => {
            require_min(index, opcode, operands, 3)?;
            Shape {
                result: Some(operands[1]),
                uses: ids2(operands[0], operands[2]),
                tag: TagSpec::OfType(operands[0]),
                ..plain()
            }
        }
        128 | 132 | 176 | 194 | 199 | 129 | 133 => {
            require_exact(index, opcode, operands, 4)?;
            Shape {
                result: Some(operands[1]),
                uses: ids3(operands[0], operands[2], operands[3]),
                tag: TagSpec::OfType(operands[0]),
                ..plain()
            }
        }
        169 => {
            require_exact(index, opcode, operands, 5)?;
            Shape {
                result: Some(operands[1]),
                uses: ids4(operands[0], operands[2], operands[3], operands[4]),
                tag: TagSpec::OfType(operands[0]),
                ..plain()
            }
        }
        4452 => {
            require_exact(index, opcode, operands, 5)?;
            if operands[4] != 0 {
                return Err(CheckError::BadPackedFormat {
                    index,
                    format: operands[4],
                });
            }
            Shape {
                result: Some(operands[1]),
                uses: ids3(operands[0], operands[2], operands[3]),
                tag: TagSpec::OfType(operands[0]),
                sudot: Some((operands[0], operands[2], operands[3])),
                ..plain()
            }
        }
        246 => {
            require_exact(index, opcode, operands, 3)?;
            plain()
        }
        248 => {
            require_exact(index, opcode, operands, 1)?;
            Shape {
                result: Some(operands[0]),
                ..plain()
            }
        }
        250 => {
            require_exact(index, opcode, operands, 3)?;
            Shape {
                uses: ids1(operands[0]),
                ..plain()
            }
        }
        _ => return Err(CheckError::MalformedInstruction { index, opcode }),
    };
    Ok(shape)
}

/// Apply the D2-D4 obligations of a decoded shape and extend the scan state.
fn apply_shape(
    index: usize,
    shape: &Shape,
    defs: &mut Vec<u32>,
    tys: &mut Vec<(u32, TyTag)>,
) -> Result<(), CheckError> {
    if let Some(storage_class) = shape.storage_class {
        if !storage_class_admitted(storage_class) {
            return Err(CheckError::BadStorageClass {
                index,
                storage_class,
            });
        }
    }
    if let Some(decoration) = shape.decoration {
        if !decoration_admitted(decoration) {
            return Err(CheckError::BadDecoration { index, decoration });
        }
    }
    if let Some((ty, a, b)) = shape.sudot {
        let result_ok = match ty_get(tys, ty) {
            Some(TyTag::Int(width)) => width >= 32,
            _ => false,
        };
        let operands_ok = matches!(ty_get(tys, a), Some(TyTag::Int(32)))
            && matches!(ty_get(tys, b), Some(TyTag::Int(32)));
        if !(result_ok && operands_ok) {
            return Err(CheckError::TypeMismatch { index });
        }
    }
    let mut u: usize = 0;
    while u < shape.uses.len() {
        let used = shape.uses[u];
        if !defs_contains(defs, used) {
            return Err(CheckError::UndefinedId { index, id: used });
        }
        u += 1;
    }
    if let Some(result) = shape.result {
        if defs_contains(defs, result) {
            return Err(CheckError::DuplicateResultId { index, id: result });
        }
        defs.push(result);
        if let Some(tag) = resolve_tag(shape.tag, tys) {
            ty_insert(tys, result, tag);
        }
    }
    Ok(())
}

/// Whether `opcode` is in the fixed [`SERVED_OPCODES`] table. Indexed linear scan: the Aeneas model
/// of this extraction unit has no slice `contains`, and an `iter().find`/`iter().any` default method
/// does not translate.
fn served_contains(opcode: u32) -> bool {
    let mut i: usize = 0;
    while i < SERVED_OPCODES.len() {
        if SERVED_OPCODES[i] == opcode {
            return true;
        }
        i += 1;
    }
    false
}

/// Whether `storage_class` is in the fixed [`VALID_STORAGE_CLASSES`] table.
fn storage_class_admitted(storage_class: u32) -> bool {
    let mut i: usize = 0;
    while i < VALID_STORAGE_CLASSES.len() {
        if VALID_STORAGE_CLASSES[i] == storage_class {
            return true;
        }
        i += 1;
    }
    false
}

/// Whether `decoration` is in the fixed [`VALID_DECORATIONS`] table.
fn decoration_admitted(decoration: u32) -> bool {
    let mut i: usize = 0;
    while i < VALID_DECORATIONS.len() {
        if VALID_DECORATIONS[i] == decoration {
            return true;
        }
        i += 1;
    }
    false
}

/// Membership in the defined-id vector: one linear scan of the D2 prefix.
fn defs_contains(defs: &[u32], id: u32) -> bool {
    let mut i: usize = 0;
    while i < defs.len() {
        if defs[i] == id {
            return true;
        }
        i += 1;
    }
    false
}

/// The tag recorded for `id`, if any: one linear scan of the id-to-tag vector.
fn ty_get(tys: &[(u32, TyTag)], id: u32) -> Option<TyTag> {
    let mut i: usize = 0;
    while i < tys.len() {
        if tys[i].0 == id {
            return Some(tys[i].1);
        }
        i += 1;
    }
    None
}

/// Record `tag` as the tag of `id`, replacing an existing entry; the map insert.
fn ty_insert(tys: &mut Vec<(u32, TyTag)>, id: u32, tag: TyTag) {
    let mut i: usize = 0;
    while i < tys.len() {
        if tys[i].0 == id {
            tys[i].1 = tag;
            return;
        }
        i += 1;
    }
    tys.push((id, tag));
}

/// Resolve a `TagSpec` against the defined-type table, mirroring `Spirv.Decode.resultTag`.
#[expect(
    clippy::manual_map,
    reason = "Option::map is an unmodeled external in the Aeneas extraction, so the explicit \
              match is required for an axiom-free extracted body"
)]
fn resolve_tag(spec: TagSpec, tys: &[(u32, TyTag)]) -> Option<TyTag> {
    match spec {
        TagSpec::None => None,
        TagSpec::Void => Some(TyTag::Void),
        TagSpec::Bool => Some(TyTag::Bool),
        TagSpec::Int(width) => Some(TyTag::Int(width)),
        TagSpec::Float(width) => Some(TyTag::Float(width)),
        TagSpec::Vector(elem) => match ty_get(tys, elem) {
            Some(_) => Some(TyTag::Vec),
            None => None,
        },
        TagSpec::RuntimeArray(elem) => match ty_get(tys, elem) {
            Some(_) => Some(TyTag::RuntimeArray),
            None => None,
        },
        TagSpec::Pointer(pointee) => match ty_get(tys, pointee) {
            Some(_) => Some(TyTag::Pointer),
            None => None,
        },
        TagSpec::Function => Some(TyTag::Function),
        TagSpec::Struct => Some(TyTag::Struct),
        TagSpec::OfType(ty) => ty_get(tys, ty),
    }
}

/// Whether any word carries a zero byte, the terminator of a SPIR-V literal string. Indexed linear
/// scan; mirrors `Spirv.Decode.hasZeroByte` without an `iter().any` default method.
fn has_literal_terminator(words: &[u32]) -> bool {
    let mut i: usize = 0;
    while i < words.len() {
        let word = words[i];
        if word.is_multiple_of(256)
            || (word / 256).is_multiple_of(256)
            || (word / 65_536).is_multiple_of(256)
            || (word / 16_777_216).is_multiple_of(256)
        {
            return true;
        }
        i += 1;
    }
    false
}
