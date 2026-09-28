//! REG-1 coverage census: parse, validate and count the operation/evidence registry.
//!
//! The registry (`coverage.json`) is data, not a claim. This module parses it with a local
//! dependency-free JSON reader, validates every row against the actual source it names, and counts
//! two explicit denominators:
//!
//! - `current_api`: the admitted `apeiron::Table` surface `{new, get, insert_with, build}`.
//! - `reg1_required`: the RFC REG-1 surface
//!   `{get, insert, remove, growth, ordered_merge, commutative_merge, refusal, encode, decode}`.
//!
//! Aeneas/structural, checked refinement and executed Rust tests are counted separately. Historical
//! Kani receipts are not current evidence and are not represented as a coverage dimension. The
//! fractions are computed from the validated rows, never from a grep count.
//!
//! The controls corrupt the parsed registry in memory and require the same validator to reject the
//! corruption; no source is mutated.

#![allow(
    clippy::module_name_repetitions,
    reason = "the module owns the coverage census and its registry"
)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{print_identity, print_report, AxisRecord, AxisStatus, Report};

/// The registry embedded from the example directory.
const REGISTRY: &str = include_str!("coverage.json");
/// The generated extraction, checked for every structural declaration a row names.
const WORD_LEAN: &str = include_str!("../../proof/ApeironWord.lean");
/// The axiom audit, checked for every refinement theorem a row names.
const AXIOM_AUDIT: &str = include_str!("../../proof/corpus/AxiomAudit.lean");
/// The extraction root, checked for every wrapper a row names.
const EXTRACT_RS: &str = include_str!("../../proof/extract/lib.rs");
/// The extraction command, used as the authority for the selected public roots.
const REGEN_SH: &str = include_str!("../../proof/regen.sh");
/// The executing table source, checked for every implementation declaration a row names.
const TABLE_RS: &str = include_str!("../../src/table.rs");

const SCHEMA: &str = "apeiron.reg1.coverage.v2";
const CURRENT_API: &str = "current_api";
const REG1_REQUIRED: &str = "reg1_required";
const CURRENT_API_OPS: [&str; 4] = ["new", "get", "insert_with", "build"];
/// The RFC REG-1 operation surface. Refusal is branch evidence under insertion/build admission,
/// not an operation of its own.
const REG1_REQUIRED_OPS: [&str; 8] = [
    "get",
    "insert",
    "remove",
    "growth",
    "ordered_merge",
    "commutative_merge",
    "encode",
    "decode",
];
/// Sources that carry a theorem or harness attribution and must therefore be digest-bound.
const REQUIRED_SOURCES: [&str; 9] = [
    "src/table.rs",
    "proof/ApeironWord.lean",
    "proof/extract/lib.rs",
    "proof/regen.sh",
    "proof/corpus/AxiomAudit.lean",
    "proof/corpus/TableRefinement.lean",
    "proof/corpus/TableRefinementGeneric.lean",
    "proof/corpus/TableRefinementPhysical.lean",
    "proof/corpus/TableRefinement/PhysicalBuildGeneric.lean",
];

/// The workload label the coverage gate prints.
pub(crate) const WORKLOAD: &str = "reg1-coverage-census";

// ── Dependency-free JSON reader (the subset the registry uses) ─────────────────────────────────

#[derive(Debug, Clone)]
enum Json {
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn as_str(&self) -> Result<&str, String> {
        match self {
            Json::Str(s) => Ok(s),
            _ => Err("expected string".to_string()),
        }
    }

    fn as_arr(&self) -> Result<&[Json], String> {
        match self {
            Json::Arr(v) => Ok(v),
            _ => Err("expected array".to_string()),
        }
    }

    fn as_obj(&self) -> Result<&[(String, Json)], String> {
        match self {
            Json::Obj(v) => Ok(v),
            _ => Err("expected object".to_string()),
        }
    }
}

/// A minimal recursive-descent reader for the JSON subset the registry uses: strings, arrays and
/// objects, with no numbers, booleans or nulls. A malformed document is an error, not a default.
struct Reader<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
}

impl<'a> Reader<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            chars: text.chars().peekable(),
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.chars.peek(), Some(c) if c.is_whitespace()) {
            self.chars.next();
        }
    }

    fn parse(mut self) -> Result<Json, String> {
        self.skip_ws();
        let value = self.value()?;
        self.skip_ws();
        if self.chars.peek().is_some() {
            return Err("trailing characters after the top-level value".to_string());
        }
        Ok(value)
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.chars.peek() {
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => Ok(Json::Str(self.string()?)),
            Some(other) => Err(format!("unexpected character {other}")),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.chars.next();
        let mut fields = Vec::new();
        self.skip_ws();
        if self.chars.peek() == Some(&'}') {
            self.chars.next();
            return Ok(Json::Obj(fields));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            if self.chars.next() != Some(':') {
                return Err("expected ':' after an object key".to_string());
            }
            let value = self.value()?;
            fields.push((key, value));
            self.skip_ws();
            match self.chars.next() {
                Some(',') => (),
                Some('}') => break,
                _ => return Err("expected ',' or '}' in object".to_string()),
            }
        }
        Ok(Json::Obj(fields))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.chars.next();
        let mut items = Vec::new();
        self.skip_ws();
        if self.chars.peek() == Some(&']') {
            self.chars.next();
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_ws();
            match self.chars.next() {
                Some(',') => (),
                Some(']') => break,
                _ => return Err("expected ',' or ']' in array".to_string()),
            }
        }
        Ok(Json::Arr(items))
    }

    fn string(&mut self) -> Result<String, String> {
        if self.chars.next() != Some('"') {
            return Err("expected a string".to_string());
        }
        let mut out = String::new();
        loop {
            match self.chars.next() {
                None => return Err("unterminated string".to_string()),
                Some('"') => break,
                Some('\\') => match self.chars.next() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('b') => out.push('\u{0008}'),
                    Some('f') => out.push('\u{000C}'),
                    Some('u') => {
                        let mut code = 0u32;
                        for _ in 0..4 {
                            let digit = self.chars.next().ok_or("short unicode escape")?;
                            let value = digit.to_digit(16).ok_or("bad unicode escape")?;
                            code = code * 16 + value;
                        }
                        out.push(char::from_u32(code).ok_or("bad code point")?);
                    }
                    _ => return Err("unknown escape".to_string()),
                },
                Some(c) => out.push(c),
            }
        }
        Ok(out)
    }
}

fn field<'a>(obj: &'a [(String, Json)], key: &str) -> Result<&'a Json, String> {
    obj.iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
        .ok_or_else(|| format!("missing field {key}"))
}

fn str_field(obj: &[(String, Json)], key: &str) -> Result<String, String> {
    field(obj, key)?
        .as_str()
        .map(str::to_string)
        .map_err(|e| format!("field {key}: {e}"))
}

fn arr_field<'a>(obj: &'a [(String, Json)], key: &str) -> Result<&'a [Json], String> {
    field(obj, key)?
        .as_arr()
        .map_err(|e| format!("field {key}: {e}"))
}

fn obj_field<'a>(obj: &'a [(String, Json)], key: &str) -> Result<&'a [(String, Json)], String> {
    field(obj, key)?
        .as_obj()
        .map_err(|e| format!("field {key}: {e}"))
}

fn string_list(json: &Json) -> Result<Vec<String>, String> {
    json.as_arr()?
        .iter()
        .map(|item| item.as_str().map(str::to_string))
        .collect()
}

// ── The registry model ────────────────────────────────────────────────────────────────────────

struct Source {
    path: String,
    sha256: String,
    pin_sha256: String,
    pin_origin: String,
    pin_status: String,
}

#[derive(Clone)]
struct Freshness {
    command: String,
    status: String,
    exit_code: String,
    receipt: String,
    digest: String,
    note: String,
}

struct Exclusion {
    id: String,
    detail: String,
}

#[derive(Clone)]
struct Row {
    id: String,
    denominator: String,
    operation: String,
    status: String,
    implementation_path: String,
    implementation_decl: String,
    extraction_root: String,
    aeneas_generated: String,
    aeneas_grade: String,
    refinement_theorem: String,
    refinement_module: String,
    refinement_grade: String,
    wrapper_theorem: String,
    wrapper_module: String,
    admission_theorem: String,
    admission_module: String,
    test_evidence: String,
    test_grade: String,
    source_path: String,
    source_sha256: String,
    freshness: String,
    note: String,
}

struct Registry {
    schema: String,
    denominators: Vec<(String, Vec<String>)>,
    sources: Vec<Source>,
    freshness: Vec<Freshness>,
    exclusions: Vec<Exclusion>,
    rows: Vec<Row>,
}

fn parse_source(json: &Json) -> Result<Source, String> {
    let obj = json.as_obj()?;
    Ok(Source {
        path: str_field(obj, "path")?,
        sha256: str_field(obj, "sha256")?,
        pin_sha256: str_field(obj, "pin_sha256")?,
        pin_origin: str_field(obj, "pin_origin")?,
        pin_status: str_field(obj, "pin_status")?,
    })
}

fn parse_freshness(json: &Json) -> Result<Freshness, String> {
    let obj = json.as_obj()?;
    Ok(Freshness {
        command: str_field(obj, "command")?,
        status: str_field(obj, "status")?,
        exit_code: str_field(obj, "exit_code")?,
        receipt: str_field(obj, "receipt")?,
        digest: str_field(obj, "digest")?,
        note: str_field(obj, "note")?,
    })
}

fn parse_exclusion(json: &Json) -> Result<Exclusion, String> {
    let obj = json.as_obj()?;
    Ok(Exclusion {
        id: str_field(obj, "id")?,
        detail: str_field(obj, "detail")?,
    })
}

fn parse_row(json: &Json) -> Result<Row, String> {
    let obj = json.as_obj()?;
    Ok(Row {
        id: str_field(obj, "id")?,
        denominator: str_field(obj, "denominator")?,
        operation: str_field(obj, "operation")?,
        status: str_field(obj, "status")?,
        implementation_path: str_field(obj, "implementation_path")?,
        implementation_decl: str_field(obj, "implementation_decl")?,
        extraction_root: str_field(obj, "extraction_root")?,
        aeneas_generated: str_field(obj, "aeneas_generated")?,
        aeneas_grade: str_field(obj, "aeneas_grade")?,
        refinement_theorem: str_field(obj, "refinement_theorem")?,
        refinement_module: str_field(obj, "refinement_module")?,
        refinement_grade: str_field(obj, "refinement_grade")?,
        wrapper_theorem: str_field(obj, "wrapper_theorem")?,
        wrapper_module: str_field(obj, "wrapper_module")?,
        admission_theorem: str_field(obj, "admission_theorem")?,
        admission_module: str_field(obj, "admission_module")?,
        test_evidence: str_field(obj, "test_evidence")?,
        test_grade: str_field(obj, "test_grade")?,
        source_path: str_field(obj, "source_path")?,
        source_sha256: str_field(obj, "source_sha256")?,
        freshness: str_field(obj, "freshness")?,
        note: str_field(obj, "note")?,
    })
}

fn parse_registry(text: &str) -> Result<Registry, String> {
    let json = Reader::new(text).parse()?;
    let obj = json.as_obj()?;
    let schema = str_field(obj, "schema")?;

    let denominators_json = obj_field(obj, "denominators")?;
    let mut denominators = Vec::new();
    for (name, value) in denominators_json {
        denominators.push((name.clone(), string_list(value)?));
    }

    let sources = arr_field(obj, "sources")?
        .iter()
        .map(parse_source)
        .collect::<Result<Vec<_>, _>>()?;
    let freshness = arr_field(obj, "freshness")?
        .iter()
        .map(parse_freshness)
        .collect::<Result<Vec<_>, _>>()?;
    let exclusions = arr_field(obj, "exclusions")?
        .iter()
        .map(parse_exclusion)
        .collect::<Result<Vec<_>, _>>()?;
    let rows = arr_field(obj, "rows")?
        .iter()
        .map(parse_row)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Registry {
        schema,
        denominators,
        sources,
        freshness,
        exclusions,
        rows,
    })
}

// ── Validation ────────────────────────────────────────────────────────────────────────────────

fn expected_ops(denominator: &str) -> Option<&'static [&'static str]> {
    match denominator {
        "current_api" => Some(&CURRENT_API_OPS),
        "reg1_required" => Some(&REG1_REQUIRED_OPS),
        _ => None,
    }
}

fn manifest_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .map_err(|e| format!("sha256sum {}: {e}", path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "sha256sum {} exited {}",
            path.display(),
            output.status
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.split_whitespace()
        .next()
        .map(str::to_string)
        .ok_or_else(|| format!("sha256sum {} produced no digest", path.display()))
}

fn find_source<'a>(sources: &'a [Source], path: &str) -> Option<&'a Source> {
    sources.iter().find(|source| source.path == path)
}

fn validate_denominators(reg: &Registry) -> Result<(), String> {
    if reg.denominators.len() != 2 {
        return Err(format!(
            "expected 2 denominators, found {}",
            reg.denominators.len()
        ));
    }
    for (name, expected) in [
        (CURRENT_API, CURRENT_API_OPS.as_slice()),
        (REG1_REQUIRED, REG1_REQUIRED_OPS.as_slice()),
    ] {
        let Some(got) = reg
            .denominators
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, ops)| ops)
        else {
            return Err(format!("missing denominator {name}"));
        };
        if got.len() != expected.len() {
            return Err(format!(
                "denominator {name} has {} operations, expected {}",
                got.len(),
                expected.len()
            ));
        }
        for op in expected {
            if !got.iter().any(|candidate| candidate.as_str() == *op) {
                return Err(format!("denominator {name} is missing {op}"));
            }
        }
    }
    Ok(())
}

fn validate_sources(reg: &Registry) -> Result<(), String> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for source in &reg.sources {
        if !seen.insert(source.path.as_str()) {
            return Err(format!("duplicate source {}", source.path));
        }
        let path = manifest_path(&source.path);
        if !path.is_file() {
            return Err(format!("source {} does not exist", source.path));
        }
        if source.pin_sha256.is_empty() != source.pin_origin.is_empty() {
            return Err(format!(
                "source {} must set pin_sha256 and pin_origin together or leave both empty",
                source.path
            ));
        }
        let computed = file_sha256(&path)?;
        if computed != source.sha256 {
            return Err(format!(
                "stale source digest for {}: recorded {} computed {}",
                source.path, source.sha256, computed
            ));
        }
        let expected_pin = if source.pin_sha256.is_empty() {
            "unpinned"
        } else if source.pin_sha256 == computed {
            "matched"
        } else {
            "stale"
        };
        if source.pin_status != expected_pin {
            return Err(format!(
                "source {} pin_status {} does not match the computed pin state {expected_pin}",
                source.path, source.pin_status
            ));
        }
    }
    for required in REQUIRED_SOURCES {
        if !seen.contains(required) {
            return Err(format!(
                "attributed source {required} is not digest-bound in the registry"
            ));
        }
    }
    Ok(())
}

fn validate_freshness(reg: &Registry) -> Result<(), String> {
    for entry in &reg.freshness {
        if entry.command.is_empty() {
            return Err("freshness entry with an empty command".to_string());
        }
        if !matches!(entry.status.as_str(), "fresh" | "stale" | "unrun") {
            return Err(format!(
                "freshness status {} for {}",
                entry.status, entry.command
            ));
        }
        if entry.status == "fresh" && (entry.exit_code != "0" || entry.receipt.is_empty()) {
            return Err(format!(
                "fresh command {} has no successful receipt",
                entry.command
            ));
        }
    }
    Ok(())
}

fn validate_implementation(row: &Row) -> Result<(), String> {
    if row.implementation_path != row.source_path {
        return Err(format!(
            "row {} implementation path is not its source path",
            row.id
        ));
    }
    if row.implementation_decl != "absent" {
        if row.implementation_path != "src/table.rs" {
            return Err(format!(
                "row {} declares an implementation outside the embedded table source",
                row.id
            ));
        }
        if !TABLE_RS.contains(&row.implementation_decl) {
            return Err(format!(
                "row {} implementation declaration {} not found in src/table.rs",
                row.id, row.implementation_decl
            ));
        }
    }
    if row.extraction_root != "absent" && !EXTRACT_RS.contains(&row.extraction_root) {
        return Err(format!(
            "row {} extraction root {} not found in proof/extract/lib.rs",
            row.id, row.extraction_root
        ));
    }
    let selected = match row.id.as_str() {
        "current-api-get" | "reg1-get" => {
            Some(("word_stored_table_get_at", "word_stored_table_get_at"))
        }
        "current-api-build" | "reg1-ordered_merge" => {
            Some(("word_stored_table_build_at", "word_stored_table_build_at"))
        }
        "reg1-commutative_merge" => Some((
            "word_stored_table_build_at",
            "table.WordMerge.Insts.Apeiron_wordTableMergeI64",
        )),
        _ => None,
    };
    let selected_roots: Vec<&str> = REGEN_SH
        .lines()
        .filter_map(|line| line.trim().strip_prefix("--start-from crate::"))
        .filter_map(|rest| rest.strip_suffix(" \\"))
        .collect();
    if selected_roots.len() != 43 {
        return Err(format!(
            "proof/regen.sh selects {} roots, expected the reviewed set of 43",
            selected_roots.len()
        ));
    }
    if let Some((root, generated)) = selected {
        if row.extraction_root != root || row.aeneas_generated != generated {
            return Err(format!(
                "row {} does not name the selected current root {root} / {generated}",
                row.id
            ));
        }
        if row.refinement_grade != "absent" {
            return Err(format!(
                "row {} claims current refinement before the selected root is linked",
                row.id
            ));
        }
        if !selected_roots.contains(&root) {
            return Err(format!(
                "row {} extraction root {root} is not selected by proof/regen.sh",
                row.id
            ));
        }
    } else if row.extraction_root != "absent"
        || row.aeneas_generated != "absent"
        || row.aeneas_grade != "absent"
    {
        return Err(format!(
            "row {} attributes structural extraction without a current selected root",
            row.id
        ));
    }
    Ok(())
}

fn validate_aeneas(row: &Row) -> Result<(), String> {
    match row.aeneas_grade.as_str() {
        "structural" => {
            if row.aeneas_generated == "absent" || !WORD_LEAN.contains(&row.aeneas_generated) {
                return Err(format!(
                    "row {} structural declaration {} not found in ApeironWord.lean",
                    row.id, row.aeneas_generated
                ));
            }
        }
        "absent" => {
            if row.aeneas_generated != "absent" {
                return Err(format!(
                    "row {} names a generated declaration but grades it absent",
                    row.id
                ));
            }
        }
        other => return Err(format!("row {} has aeneas grade {other}", row.id)),
    }
    Ok(())
}

fn require_source<'a>(
    reg: &'a Registry,
    path: &str,
    row_id: &str,
    field: &str,
) -> Result<&'a Source, String> {
    find_source(&reg.sources, path).ok_or_else(|| {
        format!("row {row_id} {field} source {path} is not digest-bound in the source registry")
    })
}

/// Validate one theorem attribution: audited in `AxiomAudit.lean`, digest-bound module, and the
/// theorem name present in that module. `absent` must be consistent across theorem and module.
fn validate_theorem(
    reg: &Registry,
    row_id: &str,
    field: &str,
    theorem: &str,
    module: &str,
) -> Result<(), String> {
    if theorem == "absent" {
        if module != "absent" {
            return Err(format!(
                "row {row_id} {field} module is set without a theorem"
            ));
        }
        return Ok(());
    }
    if module == "absent" {
        return Err(format!(
            "row {row_id} {field} theorem {theorem} has no module"
        ));
    }
    if !AXIOM_AUDIT.contains(theorem) {
        return Err(format!(
            "row {row_id} {field} theorem {theorem} is not audited in AxiomAudit.lean"
        ));
    }
    require_source(reg, module, row_id, field)?;
    let path = manifest_path(module);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("row {row_id} {field} module {module}: {e}"))?;
    let short = theorem.rsplit('.').next().unwrap_or("");
    if short.is_empty() || !text.contains(short) {
        return Err(format!(
            "row {row_id} {field} theorem {theorem} not found in {module}"
        ));
    }
    Ok(())
}

fn validate_refinement(row: &Row, reg: &Registry) -> Result<(), String> {
    match row.refinement_grade.as_str() {
        "checked" => {
            if row.refinement_theorem == "absent" {
                return Err(format!(
                    "row {} claims a checked refinement without a theorem",
                    row.id
                ));
            }
            validate_theorem(
                reg,
                &row.id,
                "refinement",
                &row.refinement_theorem,
                &row.refinement_module,
            )?;
        }
        "absent" => {
            if row.refinement_theorem != "absent" || row.refinement_module != "absent" {
                return Err(format!(
                    "row {} names a refinement theorem but grades it absent",
                    row.id
                ));
            }
        }
        other => return Err(format!("row {} has refinement grade {other}", row.id)),
    }
    validate_theorem(
        reg,
        &row.id,
        "wrapper",
        &row.wrapper_theorem,
        &row.wrapper_module,
    )?;
    validate_theorem(
        reg,
        &row.id,
        "admission",
        &row.admission_theorem,
        &row.admission_module,
    )?;
    Ok(())
}

fn validate_test(row: &Row) -> Result<(), String> {
    match row.test_grade.as_str() {
        "tested" | "smoke" => {
            if row.test_evidence == "absent" {
                return Err(format!(
                    "row {} claims {} evidence without a witness",
                    row.id, row.test_grade
                ));
            }
        }
        "absent" => {
            if row.test_evidence != "absent" {
                return Err(format!(
                    "row {} names test evidence but grades it absent",
                    row.id
                ));
            }
        }
        other => return Err(format!("row {} has test grade {other}", row.id)),
    }
    Ok(())
}

fn validate_row_status(row: &Row) -> Result<(), String> {
    let aeneas = row.aeneas_grade == "structural";
    let refinement = row.refinement_grade == "checked";
    let any = aeneas || refinement || row.test_grade != "absent";
    let extracted_refined = aeneas && refinement;
    match row.status.as_str() {
        "extracted_refined" if !extracted_refined => Err(format!(
            "row {} is extracted_refined without structural Aeneas and checked refinement evidence",
            row.id
        )),
        "partial" if extracted_refined || !any => {
            Err(format!("row {} partial status is inconsistent", row.id))
        }
        "missing" if any => Err(format!("row {} is missing but carries evidence", row.id)),
        _ => Ok(()),
    }
}

fn validate_row(row: &Row, reg: &Registry) -> Result<(), String> {
    let Some(source) = find_source(&reg.sources, &row.source_path) else {
        return Err(format!(
            "row {} references unknown source {}",
            row.id, row.source_path
        ));
    };
    if row.source_sha256 != source.sha256 {
        return Err(format!(
            "row {} source digest does not match the source registry",
            row.id
        ));
    }
    if !matches!(
        row.status.as_str(),
        "extracted_refined" | "partial" | "missing"
    ) {
        return Err(format!("row {} has status {}", row.id, row.status));
    }
    if !matches!(row.freshness.as_str(), "fresh" | "stale" | "unrun") {
        return Err(format!("row {} has freshness {}", row.id, row.freshness));
    }
    validate_implementation(row)?;
    validate_aeneas(row)?;
    validate_refinement(row, reg)?;
    validate_test(row)?;
    validate_row_status(row)
}

fn validate_rows(reg: &Registry) -> Result<(), String> {
    let mut ids: BTreeSet<&str> = BTreeSet::new();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    for row in &reg.rows {
        if !ids.insert(row.id.as_str()) {
            return Err(format!("duplicate row id {}", row.id));
        }
        let Some(ops) = expected_ops(&row.denominator) else {
            return Err(format!(
                "row {} has unknown denominator {}",
                row.id, row.denominator
            ));
        };
        if !ops.contains(&row.operation.as_str()) {
            return Err(format!(
                "row {} operation {} is not in denominator {}",
                row.id, row.operation, row.denominator
            ));
        }
        if !seen.insert((row.denominator.clone(), row.operation.clone())) {
            return Err(format!(
                "duplicate row for {}/{}",
                row.denominator, row.operation
            ));
        }
        validate_row(row, reg)?;
    }
    let expected_count = CURRENT_API_OPS.len() + REG1_REQUIRED_OPS.len();
    if reg.rows.len() != expected_count {
        return Err(format!(
            "registry has {} rows, expected {expected_count}",
            reg.rows.len()
        ));
    }
    Ok(())
}

// ── Census and report ─────────────────────────────────────────────────────────────────────────

struct DenominatorCensus {
    name: String,
    total: usize,
    aeneas: usize,
    refinement: usize,
    rust_evidence: usize,
    extracted_refined: usize,
    partial: usize,
    missing: usize,
}

struct Census {
    denominators: Vec<DenominatorCensus>,
    fresh: usize,
    stale: usize,
    unrun: usize,
    pin_match: usize,
    pin_mismatch: usize,
    unpinned: usize,
    exclusions: Vec<(String, String)>,
    freshness: Vec<Freshness>,
    rows: Vec<Row>,
}

impl Census {
    fn fraction_tokens(&self) -> Vec<String> {
        let mut tokens = Vec::new();
        for denominator in &self.denominators {
            tokens.push(format!(
                "{}/aeneas={}/{}",
                denominator.name, denominator.aeneas, denominator.total
            ));
            tokens.push(format!(
                "{}/refinement={}/{}",
                denominator.name, denominator.refinement, denominator.total
            ));
            tokens.push(format!(
                "{}/rust_evidence={}/{}",
                denominator.name, denominator.rust_evidence, denominator.total
            ));
        }
        tokens
    }
}

fn build_census(reg: &Registry) -> Census {
    let mut denominators = Vec::new();
    for name in [CURRENT_API, REG1_REQUIRED] {
        let rows: Vec<&Row> = reg
            .rows
            .iter()
            .filter(|row| row.denominator == name)
            .collect();
        denominators.push(DenominatorCensus {
            name: name.to_string(),
            total: rows.len(),
            aeneas: rows
                .iter()
                .filter(|row| row.aeneas_grade == "structural")
                .count(),
            refinement: rows
                .iter()
                .filter(|row| row.refinement_grade == "checked")
                .count(),
            rust_evidence: rows.iter().filter(|row| row.test_grade != "absent").count(),
            extracted_refined: rows
                .iter()
                .filter(|row| row.status == "extracted_refined")
                .count(),
            partial: rows.iter().filter(|row| row.status == "partial").count(),
            missing: rows.iter().filter(|row| row.status == "missing").count(),
        });
    }
    let fresh = reg
        .freshness
        .iter()
        .filter(|entry| entry.status == "fresh")
        .count();
    let stale = reg
        .freshness
        .iter()
        .filter(|entry| entry.status == "stale")
        .count();
    let unrun = reg
        .freshness
        .iter()
        .filter(|entry| entry.status == "unrun")
        .count();
    let mut pin_match = 0usize;
    let mut pin_mismatch = 0usize;
    let mut unpinned = 0usize;
    for source in &reg.sources {
        if source.pin_sha256.is_empty() {
            unpinned += 1;
        } else if source.pin_sha256 == source.sha256 {
            pin_match += 1;
        } else {
            pin_mismatch += 1;
        }
    }
    let exclusions = reg
        .exclusions
        .iter()
        .map(|exclusion| (exclusion.id.clone(), exclusion.detail.clone()))
        .collect();
    Census {
        denominators,
        fresh,
        stale,
        unrun,
        pin_match,
        pin_mismatch,
        unpinned,
        exclusions,
        freshness: reg.freshness.clone(),
        rows: reg.rows.clone(),
    }
}

fn build_coverage_report(census: &Census) -> Report {
    let fractions = census.fraction_tokens().join(" ");
    let extracted_refined: usize = census
        .denominators
        .iter()
        .map(|d| d.extracted_refined)
        .sum();
    let partial: usize = census.denominators.iter().map(|d| d.partial).sum();
    let missing: usize = census.denominators.iter().map(|d| d.missing).sum();
    let total: usize = census.denominators.iter().map(|d| d.total).sum();
    let exclusion_ids: Vec<String> = census.exclusions.iter().map(|(id, _)| id.clone()).collect();
    let evidence = format!(
        "fractions: {fractions}; extracted_refined={extracted_refined}/{total} partial={partial} missing={missing}; \
         exclusions={}; freshness=fresh={},stale={},unrun={}; \
         pin=matched={},mismatched={},unpinned={}; Kani is retired and historical receipts are not \
         current evidence; REG-1/G1 stays OPEN",
        exclusion_ids.join(","),
        census.fresh,
        census.stale,
        census.unrun,
        census.pin_match,
        census.pin_mismatch,
        census.unpinned
    );
    let axes = vec![
        AxisRecord {
            axis: "identity",
            status: AxisStatus::Missing,
            evidence: "the coverage census does not rerun the identity workload; the W2 \
                       correctness smoke records the fixed-shard identity incompatibility"
                .to_string(),
        },
        AxisRecord {
            axis: "round_trip",
            status: AxisStatus::Boundary,
            evidence: "apeiron::Table exposes no encode/decode surface; the sentinel/boundary \
                       round-trip is not exercisable"
                .to_string(),
        },
        AxisRecord {
            axis: "determinism",
            status: AxisStatus::Missing,
            evidence: "the coverage census does not rerun the separate-process determinism check"
                .to_string(),
        },
        AxisRecord {
            axis: "growth_merge_refusal",
            status: AxisStatus::Boundary,
            evidence: "remove and growth are absent required operations; ShardFull refusal is the \
                       observed boundary; a coverage census does not repair this row"
                .to_string(),
        },
        AxisRecord {
            axis: "performance",
            status: AxisStatus::Missing,
            evidence: "the coverage census measures no ratio; see --check performance".to_string(),
        },
        AxisRecord {
            axis: "coverage",
            status: AxisStatus::Boundary,
            evidence,
        },
    ];
    Report {
        axes,
        report_complete: false,
        reg1_pass: false,
    }
}

fn validate_report(report: &Report, census: &Census) -> Result<(), String> {
    let names: Vec<&str> = report.axes.iter().map(|axis| axis.axis).collect();
    let expected = [
        "identity",
        "round_trip",
        "determinism",
        "growth_merge_refusal",
        "performance",
        "coverage",
    ];
    if names != expected {
        return Err(format!("axis schema mismatch: {names:?}"));
    }
    if report.axes.iter().any(|axis| !crate::allowed(axis.status)) {
        return Err("axis status outside pass/fail/boundary/missing/tool_error".to_string());
    }
    if report.axes[0].status != AxisStatus::Missing {
        return Err("identity must be missing: the census does not rerun it".to_string());
    }
    if report.axes[1].status != AxisStatus::Boundary {
        return Err("round_trip must be boundary: no codec surface".to_string());
    }
    if report.axes[2].status != AxisStatus::Missing {
        return Err("determinism must be missing: the census does not rerun it".to_string());
    }
    if report.axes[3].status != AxisStatus::Boundary {
        return Err("growth_merge_refusal must be boundary: remove/growth absent".to_string());
    }
    if report.axes[4].status != AxisStatus::Missing {
        return Err("performance must be missing: the census measures no ratio".to_string());
    }
    if report.axes[5].status != AxisStatus::Boundary {
        return Err("coverage must be boundary while required rows are absent".to_string());
    }
    let evidence = &report.axes[5].evidence;
    for token in census.fraction_tokens() {
        if !evidence.contains(&token) {
            return Err(format!("coverage evidence omits {token}"));
        }
    }
    if !evidence.contains("exclusions=") {
        return Err("coverage evidence omits exclusions".to_string());
    }
    if !evidence.contains("freshness=") {
        return Err("coverage evidence omits freshness".to_string());
    }
    if report.report_complete {
        return Err("report_complete must be false".to_string());
    }
    if report.reg1_pass {
        return Err("reg1_pass must be false: REG-1/G1 stays OPEN".to_string());
    }
    Ok(())
}

fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out
}

fn print_census(census: &Census) {
    for denominator in &census.denominators {
        println!(
            "coverage_denominator={} total={} aeneas={}/{} refinement={}/{} rust_evidence={}/{} extracted_refined={} partial={} \
             missing={}",
            denominator.name,
            denominator.total,
            denominator.aeneas,
            denominator.total,
            denominator.refinement,
            denominator.total,
            denominator.rust_evidence,
            denominator.total,
            denominator.extracted_refined,
            denominator.partial,
            denominator.missing
        );
    }
    for token in census.fraction_tokens() {
        println!("coverage_fraction={token}");
    }
    println!(
        "coverage_freshness=fresh={} stale={} unrun={}",
        census.fresh, census.stale, census.unrun
    );
    println!(
        "coverage_pin=matched={} mismatched={} unpinned={}",
        census.pin_match, census.pin_mismatch, census.unpinned
    );
    for entry in &census.freshness {
        println!(
            "coverage_command=\"{}\" status={} digest=\"{}\" note=\"{}\"",
            escape(&entry.command),
            entry.status,
            escape(&entry.digest),
            escape(&entry.note)
        );
    }
    for (id, detail) in &census.exclusions {
        println!("coverage_exclusion={id} detail=\"{}\"", escape(detail));
    }
    for row in &census.rows {
        println!(
            "coverage_row={} denominator={} operation={} status={} aeneas={} refinement={} \
             test={} freshness={} note=\"{}\"",
            row.id,
            row.denominator,
            row.operation,
            row.status,
            row.aeneas_grade,
            row.refinement_grade,
            row.test_grade,
            row.freshness,
            escape(&row.note)
        );
    }
}

// ── Gate and controls ─────────────────────────────────────────────────────────────────────────

/// The validator controls. Each corrupts the parsed registry in memory; no source is mutated.
pub(crate) const CONTROLS: [&str; 5] = [
    "stale-source-digest",
    "failed-command-as-pass",
    "test-without-witness",
    "retired-root-as-current",
    "pin-mismatch-as-fresh",
];

/// The fail-closed expected census. A deliberate evidence change must update these numbers; silent
/// drift fails the gate rather than moving a fraction.
struct Expected {
    name: &'static str,
    total: usize,
    aeneas: usize,
    refinement: usize,
    rust_evidence: usize,
    extracted_refined: usize,
    partial: usize,
    missing: usize,
}

fn validate_expected_census(census: &Census) -> Result<(), String> {
    const EXPECTED: [Expected; 2] = [
        Expected {
            name: CURRENT_API,
            total: 4,
            aeneas: 2,
            refinement: 0,
            rust_evidence: 4,
            extracted_refined: 0,
            partial: 4,
            missing: 0,
        },
        Expected {
            name: REG1_REQUIRED,
            total: 8,
            aeneas: 3,
            refinement: 0,
            rust_evidence: 4,
            extracted_refined: 0,
            partial: 4,
            missing: 4,
        },
    ];
    if census.denominators.len() != EXPECTED.len() {
        return Err(format!(
            "census has {} denominators, expected {}",
            census.denominators.len(),
            EXPECTED.len()
        ));
    }
    for (got, want) in census.denominators.iter().zip(EXPECTED.iter()) {
        if got.name != want.name
            || got.total != want.total
            || got.aeneas != want.aeneas
            || got.refinement != want.refinement
            || got.rust_evidence != want.rust_evidence
            || got.extracted_refined != want.extracted_refined
            || got.partial != want.partial
            || got.missing != want.missing
        {
            return Err(format!(
                "denominator {} census total={} aeneas={} refinement={} rust_evidence={} extracted_refined={} partial={} \
                 missing={} does not match the expected total={} aeneas={} refinement={} rust_evidence={} extracted_refined={} \
                 partial={} missing={}",
                got.name,
                got.total,
                got.aeneas,
                got.refinement,
                got.rust_evidence,
                got.extracted_refined,
                got.partial,
                got.missing,
                want.total,
                want.aeneas,
                want.refinement,
                want.rust_evidence,
                want.extracted_refined,
                want.partial,
                want.missing
            ));
        }
    }
    Ok(())
}

/// Validate the embedded registry into a census.
fn validate_registry(reg: &Registry) -> Result<Census, String> {
    if reg.schema != SCHEMA {
        return Err(format!("schema {} is not {SCHEMA}", reg.schema));
    }
    validate_denominators(reg)?;
    validate_sources(reg)?;
    validate_freshness(reg)?;
    validate_rows(reg)?;
    let census = build_census(reg);
    validate_expected_census(&census)?;
    Ok(census)
}

/// Run the coverage census, validate it, and (when requested) require every control rejected.
pub(crate) fn run_coverage(controls: bool) -> i32 {
    let registry = match parse_registry(REGISTRY) {
        Ok(registry) => registry,
        Err(e) => {
            eprintln!("coverage registry parse failed: {e}");
            return 1;
        }
    };
    let census = match validate_registry(&registry) {
        Ok(census) => census,
        Err(e) => {
            eprintln!("coverage validation failed: {e}");
            return 1;
        }
    };
    let report = build_coverage_report(&census);
    print_identity(WORKLOAD);
    print_census(&census);
    print_report(&report);
    if let Err(e) = validate_report(&report, &census) {
        eprintln!("coverage report validation failed: {e}");
        return 1;
    }
    if controls {
        for name in CONTROLS {
            match run_control(name) {
                Ok(code) => println!("control={name} observed=red child_exit={code}"),
                Err(e) => {
                    eprintln!("control {name} failed: {e}");
                    return 1;
                }
            }
        }
    }
    0
}

/// Spawn one control child and require it to be rejected (nonzero exit and a `rejected` marker).
fn run_control(name: &str) -> Result<i32, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let output = Command::new(exe)
        .args(["--coverage-control", name])
        .output()
        .map_err(|e| format!("spawn control {name}: {e}"))?;
    if output.status.success() {
        return Err(format!(
            "control {name} exited 0; corrupted registry was accepted"
        ));
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.contains("rejected") {
        return Err(format!(
            "control {name} failed for an unexpected reason: {}",
            stderr.trim()
        ));
    }
    Ok(output.status.code().unwrap_or(-1))
}

/// The control child: parse the real registry, corrupt one property in memory, and require the same
/// validator to reject it.
pub(crate) fn run_control_child(name: &str) -> i32 {
    let mut registry = match parse_registry(REGISTRY) {
        Ok(registry) => registry,
        Err(e) => {
            eprintln!("coverage-control registry parse failed: {e}");
            return 1;
        }
    };
    let applied = match name {
        "stale-source-digest" => match registry.sources.first_mut() {
            Some(source) => {
                source.sha256 = "0".repeat(64);
                true
            }
            None => false,
        },
        "failed-command-as-pass" => match registry
            .freshness
            .iter_mut()
            .find(|entry| entry.status != "fresh")
        {
            Some(entry) => {
                entry.status = "fresh".to_string();
                entry.exit_code = "1".to_string();
                entry.receipt.clear();
                true
            }
            None => false,
        },
        "test-without-witness" => match registry
            .rows
            .iter_mut()
            .find(|row| row.test_grade == "absent")
        {
            Some(row) => {
                row.test_grade = "tested".to_string();
                row.test_evidence = "absent".to_string();
                true
            }
            None => false,
        },
        "retired-root-as-current" => match registry
            .rows
            .iter_mut()
            .find(|row| row.id == "current-api-get")
        {
            Some(row) => {
                row.extraction_root = "word_table_get".to_string();
                row.aeneas_generated = "table.Table.get".to_string();
                true
            }
            None => false,
        },
        "pin-mismatch-as-fresh" => {
            match registry
                .sources
                .iter_mut()
                .find(|source| source.pin_status == "stale")
            {
                Some(source) => {
                    source.pin_status = "matched".to_string();
                    true
                }
                None => false,
            }
        }
        _ => {
            eprintln!("unknown coverage control: {name}");
            return 2;
        }
    };
    if !applied {
        eprintln!("coverage-control {name} could not be applied to the registry");
        return 1;
    }
    match validate_registry(&registry) {
        Ok(_) => {
            eprintln!("coverage-control {name} NOT rejected: validator accepted corruption");
            1
        }
        Err(e) => {
            eprintln!("coverage-control {name} rejected: {e}");
            2
        }
    }
}
