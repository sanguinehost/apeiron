//! The `utoipa` masquerade — `#[derive(ToSchema)]`/`#[derive(IntoParams)]`/
//! `#[utoipa::path(...)]`/`#[derive(OpenApi)]`, all built on the SAME universal `SyntaxTerm` +
//! `render_surface` this crate already uses for value serialization (`codec::syntax`). No new
//! type system: an OpenAPI Schema Object, Parameter Object, PathItem, and the whole document ARE
//! just JSON, and `SyntaxTerm` already describes arbitrary JSON-shaped trees — this module is
//! only the constructors that shape those trees the way the OpenAPI 3.x spec expects, plus the
//! two traits `#[derive(...)]` implements. `.to_json()` on the assembled document is exactly
//! `render_surface(&term, Surface::Json)` — the existing renderer, unchanged.
//!
//! Scope is real usage (surveyed): `#[schema(value_type = str)]` is the only field-level
//! override in use; `#[derive(OpenApi)]` aggregates one `paths(...)` list + the schemas those
//! paths and their bodies reference. Full JSON Schema features (`oneOf`/`allOf`/examples/
//! discriminators) aren't implemented because nothing here uses them.

use crate::codec::syntax::{render_surface, Atom, Surface, SyntaxTerm};

/// `#[hashtrinity::openapi::path(...)]` — the `#[utoipa::path(...)]` masquerade, re-exported
/// here (not the crate root) so a mechanical `#[utoipa::path(` -> `#[hashtrinity::openapi::path(`
/// swap is the entire migration at each of its call sites.
pub use hashtrinity_derive::path;

fn str_atom(s: impl Into<String>) -> SyntaxTerm {
    SyntaxTerm::B(Atom::Str(s.into()))
}

fn obj(fields: Vec<(&str, SyntaxTerm)>) -> SyntaxTerm {
    SyntaxTerm::Bu(
        fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

/// `{"type": "string"}`
#[must_use]
pub fn schema_string() -> SyntaxTerm {
    obj(vec![("type", str_atom("string"))])
}

/// `{"type": "boolean"}`
#[must_use]
pub fn schema_bool() -> SyntaxTerm {
    obj(vec![("type", str_atom("boolean"))])
}

/// `{"type": "integer"}`
#[must_use]
pub fn schema_integer() -> SyntaxTerm {
    obj(vec![("type", str_atom("integer"))])
}

/// `{"type": "number"}`
#[must_use]
pub fn schema_number() -> SyntaxTerm {
    obj(vec![("type", str_atom("number"))])
}

/// `{"type": "array", "items": <items>}`
#[must_use]
pub fn schema_array(items: SyntaxTerm) -> SyntaxTerm {
    obj(vec![("type", str_atom("array")), ("items", items)])
}

/// `{"$ref": "#/components/schemas/<name>"}` — a reference to a registered schema, the
/// composition mechanism every nested `ToSchema` type uses instead of inlining itself.
#[must_use]
pub fn schema_ref(name: &str) -> SyntaxTerm {
    obj(vec![(
        "$ref",
        str_atom(format!("#/components/schemas/{name}")),
    )])
}

/// `{"type": "object", "properties": {...}, "required": [...]}` (the `required` key omitted
/// when empty, matching how real OpenAPI documents render an all-optional object).
#[must_use]
pub fn schema_object(properties: Vec<(String, SyntaxTerm)>, required: Vec<String>) -> SyntaxTerm {
    let mut fields = vec![
        ("type", str_atom("object")),
        (
            "properties",
            SyntaxTerm::Bu(properties.into_iter().collect()),
        ),
    ];
    if !required.is_empty() {
        fields.push((
            "required",
            SyntaxTerm::Fourth(required.into_iter().map(str_atom).collect()),
        ));
    }
    obj(fields)
}

/// `{"type": "string", "enum": [...]}` — a unit-only enum (externally-tagged string).
#[must_use]
pub fn schema_string_enum(variants: Vec<String>) -> SyntaxTerm {
    obj(vec![
        ("type", str_atom("string")),
        (
            "enum",
            SyntaxTerm::Fourth(variants.into_iter().map(str_atom).collect()),
        ),
    ])
}

/// A type describable as an OpenAPI schema — implemented by `#[derive(ToSchema)]`.
pub trait ToSchema {
    /// This type's own Schema Object (a `SyntaxTerm` shaped per the constructors above).
    fn schema_term() -> SyntaxTerm;
    /// The name this schema is registered under in `components.schemas`, and what
    /// [`schema_ref`] points at from any type embedding this one.
    fn schema_name() -> &'static str;
}

/// `{}` — an unconstrained schema, matching any JSON value. The right shape for a field whose
/// Rust type is genuinely dynamic (`serde_json::Value`, or any other type with no fixed shape):
/// `hashtrinity` has no dependency on `serde_json` (zero-dependency by design) so there's no
/// blanket `impl ToSchema for serde_json::Value` here to derive against - hand-write the impl
/// for the containing type instead, using this for each such field. Real `utoipa` ships that
/// blanket impl because it already depends on `serde_json`; this is the equivalent WITHOUT
/// taking on that dependency.
#[must_use]
pub fn schema_any() -> SyntaxTerm {
    object_value(vec![])
}

/// A struct describable as a list of query/path Parameter Objects — implemented by
/// `#[derive(IntoParams)]`.
pub trait IntoParams {
    /// One Parameter Object (`{"name", "in", "required", "schema"}`) per field.
    fn params_terms() -> Vec<SyntaxTerm>;
}

/// One `#[utoipa::path(...)]`-annotated handler's Operation — built by the attribute macro
/// into a `fn() -> (method, path, SyntaxTerm)` the `openapi!` aggregator below collects by name.
pub type PathEntry = fn() -> (&'static str, &'static str, SyntaxTerm);

/// Assemble the final OpenAPI document from every registered path entry + every schema this
/// path set transitively references — the `#[derive(OpenApi)]` masquerade's core, minus the
/// proc-macro surface (called from the generated `ApiDoc::build()` with an explicit list rather
/// than a `#[openapi(paths(...))]` attribute scanning them itself).
#[must_use]
pub fn build_document(
    title: &str,
    version: &str,
    paths: &[PathEntry],
    schemas: Vec<(&'static str, SyntaxTerm)>,
    tags: Vec<(&'static str, &'static str)>,
) -> SyntaxTerm {
    let mut by_path: Vec<(String, Vec<(String, SyntaxTerm)>)> = Vec::new();
    for entry in paths {
        let (method, path, op) = entry();
        if let Some((_, methods)) = by_path.iter_mut().find(|(p, _)| p == path) {
            methods.push((method.to_string(), op));
        } else {
            by_path.push((path.to_string(), vec![(method.to_string(), op)]));
        }
    }

    let paths_term = SyntaxTerm::Bu(
        by_path
            .into_iter()
            .map(|(path, methods)| (path, SyntaxTerm::Bu(methods)))
            .collect(),
    );
    let schemas_term = SyntaxTerm::Bu(
        schemas
            .into_iter()
            .map(|(name, term)| (name.to_string(), term))
            .collect(),
    );
    let tags_term = SyntaxTerm::Fourth(
        tags.into_iter()
            .map(|(name, description)| {
                obj(vec![
                    ("name", str_atom(name)),
                    ("description", str_atom(description)),
                ])
            })
            .collect(),
    );

    obj(vec![
        ("openapi", str_atom("3.1.0")),
        (
            "info",
            obj(vec![
                ("title", str_atom(title)),
                ("version", str_atom(version)),
            ]),
        ),
        ("paths", paths_term),
        ("components", obj(vec![("schemas", schemas_term)])),
        ("tags", tags_term),
    ])
}

/// Set (append or override) one top-level field of an already-built object `SyntaxTerm` — the
/// escape hatch for document fields that are only known at runtime (`info.description`,
/// `externalDocs`, `servers`), so `build_document`'s signature doesn't need to grow a parameter
/// for every optional OpenAPI field, most of which nothing here sets.
#[must_use]
pub fn with_field(term: SyntaxTerm, key: &str, value: SyntaxTerm) -> SyntaxTerm {
    let SyntaxTerm::Bu(mut fields) = term else {
        return term;
    };
    if let Some(existing) = fields.iter_mut().find(|(k, _)| k == key) {
        existing.1 = value;
    } else {
        fields.push((key.to_string(), value));
    }
    SyntaxTerm::Bu(fields)
}

/// Read one top-level field back out of an object `SyntaxTerm` (owned) — pairs with
/// [`with_field`] for updating a NESTED field (`info.description`) without hand-rolling the
/// `Bu` unwrap/rewrap at each call site: `with_field(doc, "info", with_field(field(doc,"info"), "description", ...))`.
#[must_use]
pub fn get_field(term: &SyntaxTerm, key: &str) -> SyntaxTerm {
    match term {
        SyntaxTerm::Bu(fields) => fields
            .iter()
            .find(|(k, _)| k == key)
            .map_or(SyntaxTerm::Nb, |(_, v)| v.clone()),
        _ => SyntaxTerm::Nb,
    }
}

/// A JSON string value — the general-purpose `SyntaxTerm` constructor for fields
/// `build_document`/`with_field` callers set directly (`servers: [{"url": ...}]`,
/// `externalDocs: {"url": ...}`), without needing every OpenAPI object shape as its own helper.
#[must_use]
pub fn str_value(s: impl Into<String>) -> SyntaxTerm {
    str_atom(s)
}

/// A JSON object value from arbitrary key/value pairs — same rationale as [`str_value`].
#[must_use]
pub fn object_value(fields: Vec<(&str, SyntaxTerm)>) -> SyntaxTerm {
    obj(fields)
}

/// A JSON array value from arbitrary elements — same rationale as [`str_value`].
#[must_use]
pub fn array_value(items: Vec<SyntaxTerm>) -> SyntaxTerm {
    SyntaxTerm::Fourth(items)
}

/// Render an assembled document (or any schema/operation `SyntaxTerm`) to JSON — the existing
/// `codec::syntax` renderer, not a new serializer.
#[must_use]
pub fn to_json(term: &SyntaxTerm) -> String {
    render_surface(term, Surface::Json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_string_renders_as_json_object() {
        assert_eq!(to_json(&schema_string()), r#"{"type":"string"}"#);
    }

    #[test]
    fn schema_object_renders_properties_and_required() {
        let term = schema_object(
            vec![
                ("name".to_string(), schema_string()),
                ("age".to_string(), schema_integer()),
            ],
            vec!["name".to_string()],
        );
        let json = to_json(&term);
        assert!(json.contains(r#""type":"object""#));
        assert!(json.contains(r#""properties""#));
        assert!(json.contains(r#""required":["name"]"#));
    }

    #[test]
    fn schema_object_omits_required_when_all_optional() {
        let term = schema_object(vec![("name".to_string(), schema_string())], vec![]);
        assert!(!to_json(&term).contains("required"));
    }

    #[test]
    fn schema_array_wraps_items() {
        let term = schema_array(schema_string());
        let json = to_json(&term);
        assert!(json.contains(r#""type":"array""#));
        assert!(json.contains(r#""items":{"type":"string"}"#));
    }

    #[test]
    fn schema_ref_points_at_components_schemas() {
        let json = to_json(&schema_ref("ClientResponse"));
        assert_eq!(json, r##"{"$ref":"#/components/schemas/ClientResponse"}"##);
    }

    #[test]
    fn schema_any_is_an_unconstrained_object() {
        assert_eq!(to_json(&schema_any()), "{}");
    }

    #[test]
    fn build_document_groups_methods_under_shared_paths() {
        fn get_users() -> (&'static str, &'static str, SyntaxTerm) {
            (
                "get",
                "/users",
                obj(vec![("summary", str_atom("list users"))]),
            )
        }
        fn post_users() -> (&'static str, &'static str, SyntaxTerm) {
            (
                "post",
                "/users",
                obj(vec![("summary", str_atom("create user"))]),
            )
        }
        let doc = build_document("test", "1.0", &[get_users, post_users], vec![], vec![]);
        let json = to_json(&doc);
        assert!(json.contains(r#""/users":{"get":"#));
        assert!(json.contains(r#""post":"#));
    }
}
