//! The shader half of `gate onepath`: every `.comp` and `.spv` under `src/`, each `.comp`'s text,
//! whether it computes a product, and which fns embed its `.spv`.
//!
//! The working agreement's § Goal, gap and method states it: a port lands as a face of the engine's
//! one kernel for its role, never as a second kernel. The engine holds two: `coopmat_gemm.comp` for
//! a matrix product and `attn_coopmat.comp` for attention, together the pp512 1.795x llama.cpp
//! receipt. Every other `.comp` is a finding until a `[[kernel]]` row of
//! `docs/contracts/one-path.toml` names it, and a shader that computes a product is a finding
//! unless its row carries `product = true` or a `[[debt]] rule = "shader-product"` row carries it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::{Command, Stdio};

use syn::visit::Visit;

use super::load::Tree;
use super::rules::Finding;

/// The one tree the shaders live in.
const SHADERS: &str = "src";

/// The `.comp` files (path and text), the `.spv` files, and each `.spv`'s embed sites.
pub(crate) struct Inventory {
    pub comps: BTreeMap<String, String>,
    pub spvs: BTreeSet<String>,
    /// `.spv` path -> the keys of the fns whose body embeds it with `include_bytes!`.
    pub embed: BTreeMap<String, BTreeSet<String>>,
}

/// The face a `.comp` is compiled to: its path with the extension replaced.
pub(crate) fn face_of(comp: &str) -> String {
    comp.strip_suffix(".comp")
        .map(|s| format!("{s}.spv"))
        .unwrap_or_else(|| comp.to_string())
}

/// The embed sites of every shader the TREE's fns name with `include_bytes!("…")`. A relative path
/// resolves against the declaring file's directory, which is how `include_bytes!` resolves.
///
/// Cost: time O(N) over the N AST nodes visited plus O(E) over the E embed sites; space O(E).
pub(crate) fn embeds(tree: &Tree) -> BTreeMap<String, BTreeSet<String>> {
    struct Scan {
        sites: Vec<String>,
    }
    impl<'ast> Visit<'ast> for Scan {
        fn visit_macro(&mut self, m: &'ast syn::Macro) {
            let last = m
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            if last == "include_bytes" {
                for tt in m.tokens.clone() {
                    if let proc_macro2::TokenTree::Literal(lit) = tt {
                        let text = lit.to_string();
                        let path = text.trim().trim_matches('"');
                        if path.ends_with(".spv") {
                            self.sites.push(path.to_string());
                        }
                    }
                }
            }
            syn::visit::visit_macro(self, m);
        }
    }
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for f in &tree.fns {
        let mut scan = Scan { sites: Vec::new() };
        scan.visit_block(&f.body);
        for site in scan.sites {
            let dir = f.file.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
            let path = normalize(&format!("{dir}/{site}"));
            out.entry(path).or_default().insert(f.key());
        }
    }
    out
}

/// Resolves `.` and `..` segments of a repository-relative path.
///
/// Cost: time O(S) over the S path segments, space O(S).
fn normalize(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            s => out.push(s),
        }
    }
    out.join("/")
}

/// The number of fns that embed any of `faces`.
pub(crate) fn consumers(inv: &Inventory, faces: &[String]) -> usize {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for face in faces {
        if let Some(set) = inv.embed.get(face) {
            seen.extend(set.iter().map(String::as_str));
        }
    }
    seen.len()
}

/// Reads the inventory from the working tree or the git index.
///
/// Cost: one directory walk (working tree) or one `git ls-files`/`git ls-tree` plus one
/// `git cat-file --batch` (index); time O(B) over the B bytes read, space O(B).
pub(crate) fn load(root: &Path, staged: bool) -> Result<Inventory, String> {
    let paths = match staged {
        true => git_paths(root, None)?,
        false => tree_paths(root)?,
    };
    texts(root, staged, &paths)
}

/// Reads the inventory at a git revision (`HEAD`), for the ratchet.
pub(crate) fn at_rev(root: &Path, rev: &str) -> Result<Inventory, String> {
    let paths = git_paths(root, Some(rev))?;
    let mut inv = Inventory {
        comps: BTreeMap::new(),
        spvs: BTreeSet::new(),
        embed: BTreeMap::new(),
    };
    for p in &paths {
        if p.ends_with(".spv") {
            inv.spvs.insert(p.clone());
        }
    }
    let mut child = Command::new("git")
        .current_dir(root)
        .args(["cat-file", "--batch"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("git cat-file: {e}"))?;
    let comps: Vec<&String> = paths.iter().filter(|p| p.ends_with(".comp")).collect();
    let mut request = String::new();
    for p in &comps {
        request.push_str(&format!("{rev}:{p}\n"));
    }
    let guard = write_stdin(child.stdin.take(), request);
    let output = child
        .wait_with_output()
        .map_err(|e| format!("git cat-file: {e}"))?;
    guard?;
    read_batch(&output.stdout, &comps, &mut inv.comps)?;
    Ok(inv)
}

fn texts(root: &Path, staged: bool, paths: &[String]) -> Result<Inventory, String> {
    let mut inv = Inventory {
        comps: BTreeMap::new(),
        spvs: BTreeSet::new(),
        embed: BTreeMap::new(),
    };
    let comps: Vec<&String> = paths.iter().filter(|p| p.ends_with(".comp")).collect();
    for p in paths.iter().filter(|p| p.ends_with(".spv")) {
        inv.spvs.insert(p.clone());
    }
    if !staged {
        for p in &comps {
            let text =
                std::fs::read_to_string(root.join(p.as_str())).map_err(|e| format!("{p}: {e}"))?;
            inv.comps.insert((*p).clone(), text);
        }
        return Ok(inv);
    }
    let mut child = Command::new("git")
        .current_dir(root)
        .args(["cat-file", "--batch"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("git cat-file: {e}"))?;
    let mut request = String::new();
    for p in &comps {
        request.push_str(&format!(":{p}\n"));
    }
    let guard = write_stdin(child.stdin.take(), request);
    let output = child
        .wait_with_output()
        .map_err(|e| format!("git cat-file: {e}"))?;
    guard?;
    read_batch(&output.stdout, &comps, &mut inv.comps)?;
    Ok(inv)
}

fn write_stdin(
    stdin: Option<std::process::ChildStdin>,
    request: String,
) -> Result<Option<std::thread::JoinHandle<std::io::Result<()>>>, String> {
    let Some(mut stdin) = stdin else {
        return Ok(None);
    };
    let handle = std::thread::spawn(move || {
        use std::io::Write;
        stdin.write_all(request.as_bytes())
    });
    Ok(Some(handle))
}

/// Splits `git cat-file --batch` output into the requested paths' bodies.
fn read_batch(
    bytes: &[u8],
    paths: &[&String],
    out: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    let mut at = 0;
    for p in paths {
        let nl = bytes[at..]
            .iter()
            .position(|b| *b == b'\n')
            .ok_or("git cat-file: truncated header")?;
        let header = String::from_utf8_lossy(&bytes[at..at + nl]).into_owned();
        at += nl + 1;
        let size: usize = header
            .rsplit(' ')
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("git cat-file: `{header}` for {p}"))?;
        let body = bytes
            .get(at..at + size)
            .ok_or("git cat-file: truncated body")?;
        out.insert((*p).clone(), String::from_utf8_lossy(body).into_owned());
        at += size + 1;
    }
    Ok(())
}

/// Every `.comp` and `.spv` path under [`SHADERS`] in the working tree.
fn tree_paths(root: &Path) -> Result<Vec<String>, String> {
    fn walk(root: &Path, rel: &str, out: &mut Vec<String>) -> Result<(), String> {
        let Ok(entries) = std::fs::read_dir(root.join(rel)) else {
            return Ok(());
        };
        for entry in entries {
            let entry = entry.map_err(|e| format!("{rel}: {e}"))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let child = format!("{rel}/{name}");
            let kind = entry.file_type().map_err(|e| format!("{child}: {e}"))?;
            if kind.is_dir() {
                if name != "target" && !name.starts_with('.') {
                    walk(root, &child, out)?;
                }
            } else if name.ends_with(".comp") || name.ends_with(".spv") {
                out.push(child);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(root, SHADERS, &mut out)?;
    out.sort();
    Ok(out)
}

/// Every `.comp` and `.spv` path under [`SHADERS`] in the git index (`rev` None) or at `rev`.
fn git_paths(root: &Path, rev: Option<&str>) -> Result<Vec<String>, String> {
    let args: Vec<&str> = match rev {
        None => vec!["ls-files", "-z", "--cached", "--", SHADERS],
        Some(r) => vec!["ls-tree", "-r", "-z", "--name-only", r, "--", SHADERS],
    };
    let out = Command::new("git")
        .current_dir(root)
        .args(&args)
        .output()
        .map_err(|e| format!("git {}: {e}", args[0]))?;
    if !out.status.success() {
        return Err(format!("git {} exited {}", args[0], out.status));
    }
    Ok(out
        .stdout
        .split(|b| *b == 0)
        .map(|p| String::from_utf8_lossy(p).into_owned())
        .filter(|p| p.ends_with(".comp") || p.ends_with(".spv"))
        .collect())
}

// ---- the product text test ------------------------------------------------------------------------

/// The findings the inventory implies: every `.comp`, every `.spv`, and every product `.comp`.
///
/// Cost: time O(B) over the B bytes of the B comp files, space O(F) over the F findings.
pub(crate) fn findings(inv: &Inventory) -> Vec<Finding> {
    let mut out = Vec::new();
    for (path, text) in &inv.comps {
        let face = face_of(path);
        out.push(Finding {
            rule: "shader",
            items: vec![path.clone()],
            at: format!("{path}:1"),
            count: None,
            note: format!("comp, face {face}"),
        });
        if let Some((line, quote)) = product_line(text) {
            out.push(Finding {
                rule: "shader-product",
                items: vec![path.clone()],
                at: format!("{path}:{line}"),
                count: Some(consumers(inv, &[face])),
                note: quote,
            });
        }
    }
    for p in &inv.spvs {
        out.push(Finding {
            rule: "shader",
            items: vec![p.clone()],
            at: format!("{p}:1"),
            count: None,
            note: "spv face".into(),
        });
    }
    out
}

/// The first line that makes `text` a product: a `coopMatMulAdd`, an `fma(` inside a `for` body, or
/// `lvalue += a * b` inside a `for` body whose loop variable the lvalue does not carry (a
/// reduction). Comments are blanked first, so a prose mention of a product is not one.
///
/// Cost: time O(B) over the B bytes of the text and O(B²) in the worst case for the nested
/// `for`-body scan (each body rescans its text once); space O(B) for the blanked copy.
pub(crate) fn product_line(text: &str) -> Option<(usize, String)> {
    let s = blank_comments(text);
    if let Some(at) = s.find("coopMatMulAdd") {
        let (line, quote) = line_of(&s, at);
        return Some((line, format!("coopMatMulAdd: {quote}")));
    }
    for (var, start, end) in for_bodies(&s) {
        if let Some(off) = s[start..end].find("fma(") {
            let (line, quote) = line_of(&s, start + off);
            return Some((line, format!("fma in a for body: {quote}")));
        }
        let Some(var) = var else { continue };
        let mut p = start;
        while let Some(off) = s[p..end].find("+=") {
            let at = p + off;
            let lhs_start = lvalue_start(&s, at, start);
            let lhs = s[lhs_start..at].trim();
            let rhs_end = stmt_end(&s, at + 2).unwrap_or(end).min(end);
            let rhs = &s[at + 2..rhs_end];
            if plain(lhs) && !mentions(lhs, &var) && product_rhs(rhs) {
                let (line, quote) = line_of(&s, at);
                return Some((line, format!("`{lhs} += …` over `{var}`: {quote}")));
            }
            p = at + 2;
        }
    }
    None
}

/// Replaces every comment body with spaces, keeping byte offsets and line breaks.
fn blank_comments(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                out.push(b' ');
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            out.extend_from_slice(b"  ");
            i += 2;
            while i < b.len() {
                if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                    out.extend_from_slice(b"  ");
                    i += 2;
                    break;
                }
                out.push(if b[i] == b'\n' { b'\n' } else { b' ' });
                i += 1;
            }
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn line_of(s: &str, at: usize) -> (usize, String) {
    let line = s[..at].matches('\n').count() + 1;
    let start = s[..at].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let end = s[at..].find('\n').map(|i| at + i).unwrap_or(s.len());
    (line, s[start..end].trim().chars().take(120).collect())
}

/// The index just past the `open` byte's matching `close`, honouring nesting.
fn match_close(s: &str, open: usize, o: u8, c: u8) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0usize;
    let mut i = open;
    while i < b.len() {
        if b[i] == o {
            depth += 1;
        } else if b[i] == c {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// The `;` that ends the statement starting at `from`, at bracket depth zero.
fn stmt_end(s: &str, from: usize) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    for (i, &ch) in b.iter().enumerate().skip(from) {
        match ch {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
            }
            b';' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// The start of the lvalue ending at `at`: back to the nearest statement or block boundary.
fn lvalue_start(s: &str, at: usize, floor: usize) -> usize {
    let b = s.as_bytes();
    let mut i = at;
    while i > floor {
        let ch = b[i - 1];
        if ch == b';' || ch == b'{' || ch == b'}' || ch == b'\n' {
            return i;
        }
        i -= 1;
    }
    floor
}

/// Every `for (...)` body as `(declared loop variable, body start, body end)`; nested bodies are
/// reported too.
fn for_bodies(s: &str) -> Vec<(Option<String>, usize, usize)> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let word = ident_at(s, i);
        if word != Some("for") {
            i += 1;
            continue;
        }
        let mut j = i + 3;
        while j < b.len() && (b[j] as char).is_whitespace() {
            j += 1;
        }
        let Some(header_end) = (j < b.len() && b[j] == b'(').then(|| match_close(s, j, b'(', b')'))
        else {
            i += 1;
            continue;
        };
        let Some(header_end) = header_end else {
            i += 1;
            continue;
        };
        let var = loop_var(&s[j + 1..header_end]);
        let mut k = header_end + 1;
        while k < b.len() && (b[k] as char).is_whitespace() {
            k += 1;
        }
        if k < b.len() && b[k] == b'{' {
            if let Some(close) = match_close(s, k, b'{', b'}') {
                out.push((var, k + 1, close));
                i = k + 1;
                continue;
            }
        } else if let Some(close) = stmt_end(s, k) {
            out.push((var, k, close));
            i = k + 1;
            continue;
        }
        i += 1;
    }
    out
}

/// The identifier starting at `at`, if any.
fn ident_at(s: &str, at: usize) -> Option<&str> {
    let b = s.as_bytes();
    if at >= b.len() || !(b[at].is_ascii_alphabetic() || b[at] == b'_') {
        return None;
    }
    if at > 0 && (b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'_') {
        return None;
    }
    let mut e = at;
    while e < b.len() && (b[e].is_ascii_alphanumeric() || b[e] == b'_') {
        e += 1;
    }
    Some(&s[at..e])
}

/// The variable a `for` header declares: the last identifier before its first `=`.
fn loop_var(header: &str) -> Option<String> {
    let mut depth = 0i32;
    let mut cut = header.len();
    for (i, ch) in header.char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            '=' if depth == 0 => {
                cut = i;
                break;
            }
            _ => {}
        }
    }
    if cut == header.len() {
        return None;
    }
    let before = &header[..cut];
    let mut last = None;
    let mut i = 0;
    while i < before.len() {
        if let Some(w) = ident_at(before, i) {
            last = Some(w.to_string());
            i += w.len();
        } else {
            i += 1;
        }
    }
    last
}

/// An identifier with any number of `[index]` and `.member` suffixes.
fn plain(t: &str) -> bool {
    let b = t.as_bytes();
    if b.is_empty() || !(b[0].is_ascii_alphabetic() || b[0] == b'_') {
        return false;
    }
    let mut i = 0;
    while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
        i += 1;
    }
    while i < b.len() {
        if b[i] == b'[' {
            let Some(close) = match_close(t, i, b'[', b']') else {
                return false;
            };
            if !balanced(&t[i..=close]) {
                return false;
            }
            i = close + 1;
        } else if b[i] == b'.' {
            let Some(w) = ident_at(t, i + 1) else {
                return false;
            };
            i += 1 + w.len();
        } else {
            return false;
        }
    }
    true
}

/// Every bracket in `t` closes in order.
fn balanced(t: &str) -> bool {
    let mut depth = 0i32;
    for ch in t.chars() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

/// Whether `t` names `var` as an identifier.
fn mentions(t: &str, var: &str) -> bool {
    let mut i = 0;
    while i < t.len() {
        match ident_at(t, i) {
            Some(w) => {
                if w == var {
                    return true;
                }
                i += w.len();
            }
            None => i += 1,
        }
    }
    false
}

/// Whether a right-hand side is a product of two plain factors. A factor that is an index
/// expression (`a[i * 4u + j]`) or a literal breaks the split, so index arithmetic and bit packing
/// are not products.
fn product_rhs(rhs: &str) -> bool {
    let mut depth = 0i32;
    let mut factors: Vec<String> = Vec::new();
    let mut part = String::new();
    for ch in rhs.chars() {
        match ch {
            '(' | '[' | '{' => {
                depth += 1;
                part.push(ch);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                part.push(ch);
            }
            '*' if depth == 0 => factors.push(std::mem::take(&mut part)),
            '?' | '|' | '&' if depth == 0 => return false,
            '<' if depth == 0 && rhs.contains("<<") => return false,
            _ => part.push(ch),
        }
    }
    factors.push(part);
    factors.len() >= 2 && factors.iter().all(|f| plain(f.trim()))
}
