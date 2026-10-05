//! The compact surfaces over the one term: JSON (writer + parser), BAML, TOON, CSV, POML, the
//! `Segment`/`BuildNode` path tree the CSV/TOON parsers rebuild through, and the escape/attr
//! helpers those surfaces share. Entry points are re-exported by [`super`].

use super::{field, Atom, SyntaxError, SyntaxTerm};

pub(super) fn render_json(term: &SyntaxTerm) -> String {
    let mut out = String::new();
    write_json(term, &mut out);
    out
}

fn write_json(term: &SyntaxTerm, out: &mut String) {
    match term {
        SyntaxTerm::Nb | SyntaxTerm::Unknown => out.push_str("null"),
        SyntaxTerm::B(Atom::Bool(v)) => out.push_str(if *v { "true" } else { "false" }),
        SyntaxTerm::B(Atom::Num(v)) => out.push_str(v),
        SyntaxTerm::B(Atom::Str(v)) => {
            out.push('"');
            push_escaped_json(out, v);
            out.push('"');
        }
        SyntaxTerm::Bu(fields) => {
            out.push('{');
            let mut first = true;
            for (k, v) in fields {
                if !first {
                    out.push(',');
                }
                first = false;
                out.push('"');
                push_escaped_json(out, k);
                out.push_str("\":");
                write_json(v, out);
            }
            out.push('}');
        }
        SyntaxTerm::Fourth(items) => {
            out.push('[');
            let mut first = true;
            for v in items {
                if !first {
                    out.push(',');
                }
                first = false;
                write_json(v, out);
            }
            out.push(']');
        }
    }
}

pub(super) fn render_baml_term(term: &SyntaxTerm, depth: usize) -> String {
    match term {
        SyntaxTerm::Nb | SyntaxTerm::Unknown => "nb".into(),
        SyntaxTerm::B(Atom::Bool(v)) => v.to_string(),
        SyntaxTerm::B(Atom::Num(v)) => v.clone(),
        SyntaxTerm::B(Atom::Str(v)) => format!("\"{}\"", escape_json(v)),
        SyntaxTerm::Bu(fields) => {
            let pad = "  ".repeat(depth);
            let inner = "  ".repeat(depth + 1);
            let mut out = String::from("{");
            for (k, v) in fields {
                out.push('\n');
                out.push_str(&inner);
                out.push_str(k);
                out.push_str(" = ");
                out.push_str(&render_baml_term(v, depth + 1));
            }
            out.push('\n');
            out.push_str(&pad);
            out.push('}');
            out
        }
        SyntaxTerm::Fourth(items) => {
            let body = items
                .iter()
                .map(|v| render_baml_term(v, depth))
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{body}]")
        }
    }
}

pub(super) fn render_toon(term: &SyntaxTerm) -> String {
    let mut lines = Vec::new();
    flatten_toon("$", term, &mut lines);
    lines.join("\n") + "\n"
}

fn flatten_toon(path: &str, term: &SyntaxTerm, out: &mut Vec<String>) {
    match term {
        SyntaxTerm::Bu(fields) => {
            for (k, v) in fields {
                let next = if path == "$" {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                };
                flatten_toon(&next, v, out);
            }
        }
        SyntaxTerm::Fourth(items) => {
            for (idx, v) in items.iter().enumerate() {
                flatten_toon(&format!("{path}[{idx}]"), v, out);
            }
        }
        _ => out.push(format!("{path} = {}", scalar_lit(term))),
    }
}

pub(super) fn render_csv(term: &SyntaxTerm) -> String {
    let mut rows = Vec::new();
    rows.push(csv_row(["path", "kind", "value"]));
    flatten_csv("$", term, &mut rows);
    rows.join("\n") + "\n"
}

fn flatten_csv(path: &str, term: &SyntaxTerm, out: &mut Vec<String>) {
    match term {
        SyntaxTerm::Nb => out.push(csv_row([path, "nb", ""])),
        SyntaxTerm::Unknown => out.push(csv_row([path, "unknown", ""])),
        SyntaxTerm::B(Atom::Bool(v)) => {
            out.push(csv_row([path, "bool", if *v { "true" } else { "false" }]));
        }
        SyntaxTerm::B(Atom::Num(v)) => out.push(csv_row([path, "num", v.as_str()])),
        SyntaxTerm::B(Atom::Str(v)) => out.push(csv_row([path, "str", v.as_str()])),
        SyntaxTerm::Bu(fields) => {
            out.push(csv_row([path, "bu", ""]));
            for (k, v) in fields {
                let next = if path == "$" {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                };
                flatten_csv(&next, v, out);
            }
        }
        SyntaxTerm::Fourth(items) => {
            out.push(csv_row([path, "fourth", ""]));
            for (idx, v) in items.iter().enumerate() {
                let next = if path == "$" {
                    format!("[{idx}]")
                } else {
                    format!("{path}[{idx}]")
                };
                flatten_csv(&next, v, out);
            }
        }
    }
}

fn csv_row<I, S>(cols: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    cols.into_iter()
        .map(|s| csv_escape(s.as_ref()))
        .collect::<Vec<_>>()
        .join(",")
}

fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

pub(super) fn parse_csv(input: &str) -> Result<SyntaxTerm, SyntaxError> {
    let mut rows = input.lines().filter(|line| !line.trim().is_empty());
    let header = rows
        .next()
        .ok_or_else(|| SyntaxError::new("csv requires a header"))?;
    let expected = ["path", "kind", "value"];
    let parsed_header = parse_csv_row(header)?;
    if parsed_header != expected {
        return Err(SyntaxError::new("csv header must be path,kind,value"));
    }
    let mut root = BuildNode::Empty;
    for row in rows {
        let cols = parse_csv_row(row)?;
        if cols.len() != 3 {
            return Err(SyntaxError::new("csv row must have exactly 3 columns"));
        }
        let path = cols[0].trim();
        let kind = cols[1].trim();
        let value = cols[2].trim();
        let term = match kind {
            "nb" => SyntaxTerm::Nb,
            "unknown" => SyntaxTerm::Unknown,
            "bool" => SyntaxTerm::B(Atom::Bool(match value {
                "true" => true,
                "false" => false,
                _ => return Err(SyntaxError::new("csv bool must be true or false")),
            })),
            "num" => SyntaxTerm::B(Atom::Num(value.to_string())),
            "str" => SyntaxTerm::B(Atom::Str(value.to_string())),
            "bu" => SyntaxTerm::Bu(Vec::new()),
            "fourth" => SyntaxTerm::Fourth(Vec::new()),
            other => return Err(SyntaxError::new(format!("unknown csv kind: {other}"))),
        };
        insert_csv_node(&mut root, path, term)?;
    }
    Ok(root.finish())
}

fn parse_csv_row(row: &str) -> Result<Vec<String>, SyntaxError> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut chars = row.chars().peekable();
    let mut quoted = false;
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted => {
                if matches!(chars.peek(), Some('"')) {
                    chars.next();
                    cur.push('"');
                } else {
                    quoted = false;
                }
            }
            '"' if cur.is_empty() => quoted = true,
            ',' if !quoted => {
                fields.push(std::mem::take(&mut cur));
            }
            _ => cur.push(c),
        }
    }
    if quoted {
        return Err(SyntaxError::new("unterminated csv quote"));
    }
    fields.push(cur);
    Ok(fields)
}

fn insert_csv_node(root: &mut BuildNode, path: &str, term: SyntaxTerm) -> Result<(), SyntaxError> {
    if path == "$" {
        *root = match term {
            SyntaxTerm::Bu(_) => BuildNode::Obj(Vec::new()),
            SyntaxTerm::Fourth(_) => BuildNode::List(Vec::new()),
            other => BuildNode::Scalar(other),
        };
        return Ok(());
    }
    let segs = parse_path(path)?;
    root.insert(&segs, term)
}

fn scalar_lit(term: &SyntaxTerm) -> String {
    match term {
        SyntaxTerm::Nb | SyntaxTerm::Unknown => "nb".into(),
        SyntaxTerm::B(Atom::Bool(v)) => v.to_string(),
        SyntaxTerm::B(Atom::Num(v)) => v.clone(),
        SyntaxTerm::B(Atom::Str(v)) => format!("\"{}\"", escape_json(v)),
        SyntaxTerm::Bu(_) | SyntaxTerm::Fourth(_) => render_json(term),
    }
}

pub(super) fn render_poml(term: &SyntaxTerm) -> String {
    format!("<poml>{}</poml>\n", render_poml_node(term))
}

fn render_poml_node(term: &SyntaxTerm) -> String {
    match term {
        SyntaxTerm::Nb | SyntaxTerm::Unknown => "<nb/>".into(),
        SyntaxTerm::B(Atom::Bool(v)) => format!("<b kind=\"bool\">{v}</b>"),
        SyntaxTerm::B(Atom::Num(v)) => format!("<b kind=\"num\">{}</b>", escape_xml(v)),
        SyntaxTerm::B(Atom::Str(v)) => format!("<b kind=\"str\">{}</b>", escape_xml(v)),
        SyntaxTerm::Bu(fields) => {
            let mut out = String::from("<bu>");
            for (k, v) in fields {
                out.push_str("<field name=\"");
                out.push_str(&escape_xml(k));
                out.push_str("\">");
                out.push_str(&render_poml_node(v));
                out.push_str("</field>");
            }
            out.push_str("</bu>");
            out
        }
        SyntaxTerm::Fourth(items) => {
            let mut out = String::from("<list>");
            for item in items {
                out.push_str("<item>");
                out.push_str(&render_poml_node(item));
                out.push_str("</item>");
            }
            out.push_str("</list>");
            out
        }
    }
}

pub(super) fn parse_baml(input: &str) -> Result<SyntaxTerm, SyntaxError> {
    let trimmed = input.trim();
    let body = trimmed.strip_prefix("baml").unwrap_or(trimmed).trim();
    JsonParser::new(body, true).parse()
}

pub(super) fn parse_toon(input: &str) -> Result<SyntaxTerm, SyntaxError> {
    let mut root = BuildNode::Obj(Vec::new());
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (path, value) = line
            .split_once('=')
            .ok_or_else(|| SyntaxError::new(format!("bad TOON line: {line}")))?;
        let segs = parse_path(path.trim())?;
        let term = JsonParser::new(value.trim(), true).parse()?;
        root.insert(&segs, term)?;
    }
    Ok(root.finish())
}

pub(super) fn parse_poml(input: &str) -> Result<SyntaxTerm, SyntaxError> {
    let body = input
        .trim()
        .strip_prefix("<poml>")
        .and_then(|s| s.strip_suffix("</poml>"))
        .ok_or_else(|| SyntaxError::new("POML must be wrapped in <poml>"))?;
    parse_poml_node(body.trim())
}

fn parse_poml_node(input: &str) -> Result<SyntaxTerm, SyntaxError> {
    let input = input.trim();
    if input == "<nb/>" {
        return Ok(SyntaxTerm::Nb);
    }
    if let Some(inner) = input
        .strip_prefix("<bu>")
        .and_then(|s| s.strip_suffix("</bu>"))
    {
        let mut fields = Vec::new();
        let mut rest = inner.trim();
        while !rest.is_empty() {
            let tag_end = rest
                .find('>')
                .ok_or_else(|| SyntaxError::new("bad POML field"))?;
            let tag = &rest[..=tag_end];
            let name = attr(tag, "name").ok_or_else(|| SyntaxError::new("field without name"))?;
            let close = "</field>";
            let end = find_balanced_close(&rest[tag_end + 1..], "<field ", close)
                .ok_or_else(|| SyntaxError::new("unclosed field"))?
                + tag_end
                + 1;
            fields.push((
                unescape_xml(&name),
                parse_poml_node(&rest[tag_end + 1..end])?,
            ));
            rest = rest[end + close.len()..].trim();
        }
        return Ok(SyntaxTerm::Bu(fields));
    }
    if let Some(inner) = input
        .strip_prefix("<list>")
        .and_then(|s| s.strip_suffix("</list>"))
    {
        let mut items = Vec::new();
        let mut rest = inner.trim();
        while !rest.is_empty() {
            let tag = "<item>";
            let close = "</item>";
            let body = rest
                .strip_prefix(tag)
                .ok_or_else(|| SyntaxError::new("bad list item"))?;
            let end = find_balanced_close(body, tag, close)
                .ok_or_else(|| SyntaxError::new("unclosed item"))?;
            items.push(parse_poml_node(&body[..end])?);
            rest = body[end + close.len()..].trim();
        }
        return Ok(SyntaxTerm::Fourth(items));
    }
    if input.starts_with("<b ") {
        let tag_end = input
            .find('>')
            .ok_or_else(|| SyntaxError::new("bad scalar tag"))?;
        let tag = &input[..=tag_end];
        let kind = attr(tag, "kind").ok_or_else(|| SyntaxError::new("scalar without kind"))?;
        let value = input[tag_end + 1..]
            .strip_suffix("</b>")
            .ok_or_else(|| SyntaxError::new("unclosed scalar"))?;
        let value = unescape_xml(value);
        return Ok(SyntaxTerm::B(match kind.as_str() {
            "bool" => Atom::Bool(value == "true"),
            "num" => Atom::Num(value),
            _ => Atom::Str(value),
        }));
    }
    Err(SyntaxError::new("unknown POML node"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Segment {
    Key(String),
    Index(usize),
}

#[derive(Clone, Debug)]
enum BuildNode {
    Empty,
    Scalar(SyntaxTerm),
    Obj(Vec<(String, BuildNode)>),
    List(Vec<BuildNode>),
}

impl BuildNode {
    fn insert(&mut self, segs: &[Segment], value: SyntaxTerm) -> Result<(), SyntaxError> {
        if segs.is_empty() {
            *self = match value {
                SyntaxTerm::Bu(_) => BuildNode::Obj(Vec::new()),
                SyntaxTerm::Fourth(_) => BuildNode::List(Vec::new()),
                other => BuildNode::Scalar(other),
            };
            return Ok(());
        }
        match &segs[0] {
            Segment::Key(key) => {
                if matches!(self, BuildNode::Empty) {
                    *self = BuildNode::Obj(Vec::new());
                }
                let BuildNode::Obj(fields) = self else {
                    return Err(SyntaxError::new("path crosses non-object"));
                };
                let pos = fields.iter().position(|(k, _)| k == key);
                let idx = if let Some(i) = pos {
                    i
                } else {
                    fields.push((key.clone(), BuildNode::Empty));
                    fields.len() - 1
                };
                fields[idx].1.insert(&segs[1..], value)
            }
            Segment::Index(i) => {
                if matches!(self, BuildNode::Empty) {
                    *self = BuildNode::List(Vec::new());
                }
                let BuildNode::List(items) = self else {
                    return Err(SyntaxError::new("path crosses non-list"));
                };
                while items.len() <= *i {
                    items.push(BuildNode::Empty);
                }
                items[*i].insert(&segs[1..], value)
            }
        }
    }

    fn finish(self) -> SyntaxTerm {
        match self {
            BuildNode::Empty => SyntaxTerm::Nb,
            BuildNode::Scalar(term) => term,
            BuildNode::Obj(fields) => {
                SyntaxTerm::Bu(fields.into_iter().map(|(k, v)| (k, v.finish())).collect())
            }
            BuildNode::List(items) => {
                SyntaxTerm::Fourth(items.into_iter().map(BuildNode::finish).collect())
            }
        }
    }
}

fn parse_path(path: &str) -> Result<Vec<Segment>, SyntaxError> {
    if path == "$" {
        return Ok(Vec::new());
    }
    let mut segs = Vec::new();
    for part in path.split('.') {
        let mut rem = part;
        if let Some(key_end) = rem.find('[') {
            if key_end > 0 {
                segs.push(Segment::Key(rem[..key_end].to_string()));
            }
            rem = &rem[key_end..];
        } else if !rem.is_empty() {
            segs.push(Segment::Key(rem.to_string()));
            continue;
        }
        while let Some(body) = rem.strip_prefix('[') {
            let end = body
                .find(']')
                .ok_or_else(|| SyntaxError::new("bad index path"))?;
            segs.push(Segment::Index(
                body[..end]
                    .parse()
                    .map_err(|_| SyntaxError::new("bad index"))?,
            ));
            rem = &body[end + 1..];
        }
        if !rem.is_empty() {
            return Err(SyntaxError::new("bad path tail"));
        }
    }
    Ok(segs)
}

pub(super) struct JsonParser<'a> {
    s: &'a [u8],
    pos: usize,
    loose: bool,
}

impl<'a> JsonParser<'a> {
    pub(super) fn new(input: &'a str, loose: bool) -> Self {
        Self {
            s: input.as_bytes(),
            pos: 0,
            loose,
        }
    }

    pub(super) fn parse(mut self) -> Result<SyntaxTerm, SyntaxError> {
        let v = self.value()?;
        self.ws();
        if self.pos != self.s.len() {
            return Err(SyntaxError::new("trailing input"));
        }
        Ok(v)
    }

    fn value(&mut self) -> Result<SyntaxTerm, SyntaxError> {
        self.ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(|s| SyntaxTerm::B(Atom::Str(s))),
            Some(b'-' | b'0'..=b'9') => Ok(SyntaxTerm::B(Atom::Num(self.number()?))),
            Some(_) if self.loose => {
                let word = self.word();
                Ok(match word.as_str() {
                    "nb" | "null" => SyntaxTerm::Nb,
                    "true" => SyntaxTerm::B(Atom::Bool(true)),
                    "false" => SyntaxTerm::B(Atom::Bool(false)),
                    _ => SyntaxTerm::B(Atom::Str(word)),
                })
            }
            _ if self.take_lit("null") => Ok(SyntaxTerm::Nb),
            _ if self.take_lit("true") => Ok(SyntaxTerm::B(Atom::Bool(true))),
            _ if self.take_lit("false") => Ok(SyntaxTerm::B(Atom::Bool(false))),
            _ => Err(SyntaxError::new("expected value")),
        }
    }

    fn object(&mut self) -> Result<SyntaxTerm, SyntaxError> {
        self.expect(b'{')?;
        let mut fields = Vec::new();
        loop {
            self.ws();
            if self.take(b'}') {
                break;
            }
            let key = if self.peek() == Some(b'"') {
                self.string()?
            } else if self.loose {
                self.word()
            } else {
                return Err(SyntaxError::new("expected quoted object key"));
            };
            self.ws();
            if !(self.take(b':') || (self.loose && self.take(b'='))) {
                return Err(SyntaxError::new("expected object separator"));
            }
            let val = self.value()?;
            fields.push((key, val));
            self.ws();
            if self.take(b'}') {
                break;
            }
            if !(self.take(b',') || (self.loose && self.take(b';'))) {
                self.ws();
                if !self.loose {
                    return Err(SyntaxError::new("expected comma"));
                }
            }
        }
        Ok(SyntaxTerm::Bu(fields))
    }

    fn array(&mut self) -> Result<SyntaxTerm, SyntaxError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        loop {
            self.ws();
            if self.take(b']') {
                break;
            }
            items.push(self.value()?);
            self.ws();
            if self.take(b']') {
                break;
            }
            self.expect(b',')?;
        }
        Ok(SyntaxTerm::Fourth(items))
    }

    /// A JSON string. Bytes are collected raw (not widened one-at-a-time to `char`, which mangled
    /// every multi-byte UTF-8 sequence — the earlier form treated a Chinese character's 3 continuation
    /// bytes as 3 separate Latin-1 codepoints) and decoded once at the end; `\uXXXX` is handled,
    /// including surrogate pairs (Python's `json.dump` default `ensure_ascii=True` encodes every
    /// non-ASCII character this way, so any CJK-bearing corpus hits this path on nearly every string).
    ///
    /// **There is deliberately no bytes-as-chars variant of this**, and a byte-level tokenizer
    /// loader is not a reason to add one. GPT-2 style vocabularies do spell each byte as its own
    /// printable char, but that alphabet (`bytes_to_unicode`) is 256 real Unicode *scalars* — `Ġ` is
    /// U+0120, not the byte pair C4 A0 — so decoding the file correctly is already exactly what such
    /// a loader wants. A parser that widened bytes only forced the loader to undo the widening, and
    /// that repair is not idempotent: run it over correct scalars and `Ã©` (the spelling of `é`'s two
    /// UTF-8 bytes) collapses back to one char. One decode, here, is the whole contract.
    fn string(&mut self) -> Result<String, SyntaxError> {
        self.expect(b'"')?;
        let mut out: Vec<u8> = Vec::new();
        while let Some(b) = self.next() {
            match b {
                b'"' => {
                    return String::from_utf8(out)
                        .map_err(|_| SyntaxError::new("bad utf8 in string"))
                }
                b'\\' => {
                    let esc = self.next().ok_or_else(|| SyntaxError::new("bad escape"))?;
                    match esc {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0C),
                        b'u' => {
                            let cp = self.hex4()?;
                            let ch = if (0xD800..=0xDBFF).contains(&cp) {
                                if self.next() != Some(b'\\') || self.next() != Some(b'u') {
                                    return Err(SyntaxError::new("unpaired surrogate"));
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&lo) {
                                    return Err(SyntaxError::new("unpaired surrogate"));
                                }
                                let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                char::from_u32(c)
                                    .ok_or_else(|| SyntaxError::new("bad codepoint"))?
                            } else {
                                char::from_u32(cp)
                                    .ok_or_else(|| SyntaxError::new("bad codepoint"))?
                            };
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        other => out.push(other),
                    }
                }
                other => out.push(other),
            }
        }
        Err(SyntaxError::new("unterminated string"))
    }

    /// Four hex digits after `\u` — the codepoint (or surrogate half) they encode.
    fn hex4(&mut self) -> Result<u32, SyntaxError> {
        let mut v = 0u32;
        for _ in 0..4 {
            let b = self
                .next()
                .ok_or_else(|| SyntaxError::new("bad \\u escape"))?;
            let d = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                _ => return Err(SyntaxError::new("bad \\u escape")),
            };
            v = v * 16 + d as u32;
        }
        Ok(v)
    }

    fn number(&mut self) -> Result<String, SyntaxError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let digit_start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        if self.pos == digit_start {
            return Err(SyntaxError::new("bad number"));
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            let frac_start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == frac_start {
                return Err(SyntaxError::new("bad number"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let exp_start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == exp_start {
                return Err(SyntaxError::new("bad number"));
            }
        }
        String::from_utf8(self.s[start..self.pos].to_vec())
            .map_err(|_| SyntaxError::new("bad number"))
    }

    fn word(&mut self) -> String {
        self.ws();
        let start = self.pos;
        while matches!(
            self.peek(),
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'-' | b'/' | b'.')
        ) {
            self.pos += 1;
        }
        String::from_utf8_lossy(&self.s[start..self.pos]).to_string()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }

    fn take_lit(&mut self, lit: &str) -> bool {
        if self.s[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), SyntaxError> {
        if self.take(b) {
            Ok(())
        } else {
            Err(SyntaxError::new("unexpected byte"))
        }
    }

    fn take(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn next(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }
}

fn attr(tag: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = tag.find(&needle)? + needle.len();
    let end = tag[start..].find('"')?;
    Some(tag[start..start + end].to_string())
}

fn find_balanced_close(s: &str, open: &str, close: &str) -> Option<usize> {
    let mut cursor = 0usize;
    let mut depth = 0usize;
    loop {
        let next_open = s[cursor..].find(open).map(|i| cursor + i);
        let next_close = s[cursor..].find(close).map(|i| cursor + i);
        match (next_open, next_close) {
            (_, Some(c)) if next_open.map_or(true, |o| c < o) => {
                if depth == 0 {
                    return Some(c);
                }
                depth -= 1;
                cursor = c + close.len();
            }
            (Some(o), _) => {
                depth += 1;
                cursor = o + open.len();
            }
            _ => return None,
        }
    }
}

fn push_escaped_json(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c => out.push(c),
        }
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    push_escaped_json(&mut out, s);
    out
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn unescape_xml(s: &str) -> String {
    s.replace("&quot;", "\"")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}
