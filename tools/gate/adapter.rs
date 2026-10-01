//! Impure boundary for `tools/gate`: filesystem, hashing, subprocess, wall clock and locks.
//!
//! `core.rs` is the pure admission decision. This file supplies what a pure body cannot have, and
//! nothing else: it reads a file, hashes it, runs a command under a declared wall-clock budget and
//! takes a lock file. It formats what it observed and decides nothing about admission; every
//! verdict comes from `core::validate` or `core::admit`. The split mirrors
//! `tools/dispatch/{core.rs,main.rs}`.
//!
//! **Trust boundary.** The command string handed to [`run_budgeted`] is executed by `/bin/sh -c`
//! without sanitising, because the registry records the *exact* command a gate must run. The
//! registry in `docs/contracts/artifacts.toml` is therefore trusted input: a row that can write a
//! command can run it. That is the same trust the manifest already claims, and it is why the
//! manifest is a `docs/contracts/` artefact and is edited under a task, never by a worker.
//!
//! **Digest rule**, identical to the header of `docs/contracts/artifacts.toml` and to the only
//! implementation here, so a recorded `sha256` is reproducible from the tree alone:
//!
//! - a file: the sha256 of its bytes, lowercase hex, no `sha256:` prefix;
//! - a directory: sort every contained file by its repository-relative path, then sha256 the
//!   concatenation of one `<hex><space><space><repo-relative path><newline>` line per file. Path
//!   components beginning with `.` and the component `target` are skipped, so build products and
//!   editor scratch do not enter an identity.
//!
//! **Complexity.** [`sha256_path`] on a file is O(bytes) time and O(1) space beyond the 64-byte
//! padding block: one pass, no buffering of the payload. On a directory it is O(files + bytes) time
//! and O(files) space in the sorted path list; the sort is over the collected names, and the
//! collection is one directory walk that terminates because the walk is over a finite tree.
//! [`run_budgeted`] is O(1) space and its wall time is bounded by the passed `budget_s`: the child
//! is polled every [`POLL_MS`] and killed on expiry, so the call returns within
//! `budget_s + POLL_MS`.

use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Poll interval for a budgeted child. The overshoot past the declared budget is bounded by this.
const POLL_MS: u64 = 25;

/// Retry interval while another process holds the lock file.
const LOCK_POLL_MS: u64 = 100;

/// Largest budget `docs/contracts/artifacts.toml` may declare. `AGENTS.md` §10 fails a local
/// invocation that reaches ten minutes, so a declared budget above that could never be met.
pub const MAX_BUDGET_S: u64 = 600;

/// `kind` values the registry schema admits.
const KINDS: [&str; 5] = ["gate", "receipt", "doc", "lean", "rust"];

// -- rows -------------------------------------------------------------------------------

/// One registry row, as parsed. A `sha256` here is the *recorded* identity; the identity on disk is
/// computed by [`sha256_path`] and compared by `core::admit`, which owns that verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub id: String,
    pub kind: String,
    pub path: String,
    pub sha256: String,
    pub gate: String,
    pub control: String,
    pub budget_s: u64,
}

/// A syntax error in the manifest, with the 1-based line it was found on. A manifest this module
/// cannot parse is not a malformed decision, so the loader reports it as an error and the caller
/// refuses to check anything; it never repairs or ignores a row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

fn err<T>(line: usize, message: impl Into<String>) -> Result<T, SyntaxError> {
    Err(SyntaxError {
        line,
        message: message.into(),
    })
}

/// The string value of a TOML basic string on one line. Only `\"` and `\\` are recognised, which
/// covers every value the schema admits; a backslash before anything else is a syntax error rather
/// than a silent reinterpretation.
fn string_value(raw: &str, line: usize) -> Result<String, SyntaxError> {
    let body = raw
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .ok_or_else(|| SyntaxError {
            line,
            message: format!("value is not a quoted string: {raw}"),
        })?;
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            if ch == '"' {
                return err(line, "unescaped quote inside a string value");
            }
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some(other) => return err(line, format!("unsupported escape `\\{other}`")),
            None => return err(line, "string value ends in a backslash"),
        }
    }
    Ok(out)
}

/// The schema's key names, returned as literals so a stored key carries no borrow of the line it
/// was read from. `None` for a key the schema does not declare, which is an unknown key and a
/// refusal: an unread field is not a field with a default.
fn canonical_key(key: &str) -> Option<&'static str> {
    match key {
        "id" => Some("id"),
        "kind" => Some("kind"),
        "path" => Some("path"),
        "sha256" => Some("sha256"),
        "gate" => Some("gate"),
        "control" => Some("control"),
        "budget_s" => Some("budget_s"),
        _ => None,
    }
}

/// A lowercase hex sha256 of exactly 64 digits, or a syntax error. A digest that is not 64 hex
/// digits cannot be compared against a computed one, so it is rejected at the syntax layer rather
/// than at the comparison.
fn digest_value(raw: &str, line: usize) -> Result<String, SyntaxError> {
    if raw.len() != 64
        || !raw
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return err(
            line,
            format!("sha256 is not 64 lowercase hex digits: {raw}"),
        );
    }
    Ok(raw.to_string())
}

/// Parse the registry. Strict TOML subset: `[[artifact]]` headers, `key = "string"` or
/// `key = <int>`, `#` comments. An unknown key, a repeated key within a row, a missing required
/// key, a `kind` outside the schema or a budget outside `[1, MAX_BUDGET_S]` is refused; nothing is
/// defaulted and nothing is skipped.
///
/// Time O(lines), space O(rows). One pass over the file; the row is completed at the next header or
/// at end of input, so no backtracking and no second parse.
pub fn parse(text: &str) -> Result<Vec<Row>, SyntaxError> {
    let mut rows: Vec<Row> = Vec::new();
    let mut fields: Vec<(&str, String)> = Vec::new();
    let mut row_line = 0usize;

    for (index, raw_line) in text.lines().enumerate() {
        let line = index + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix('[') {
            let header = rest.strip_suffix(']').ok_or_else(|| SyntaxError {
                line,
                message: format!("unterminated table header: {trimmed}"),
            })?;
            if header.trim() != "[artifact]" {
                return err(line, format!("unknown table header: [{header}]"));
            }
            if !fields.is_empty() {
                finish_row(&mut fields, &mut rows, row_line)?;
            }
            row_line = line;
            continue;
        }

        if row_line == 0 {
            return err(line, "a key appears before the first [[artifact]] header");
        }

        let (key, value) = trimmed.split_once('=').ok_or_else(|| SyntaxError {
            line,
            message: format!("not a `key = value` line: {trimmed}"),
        })?;
        let key = key.trim();
        let value = value.trim();
        if fields.iter().any(|(seen, _)| *seen == key) {
            return err(line, format!("key `{key}` repeated within one row"));
        }
        // The canonical name, taken from the pattern and not from the slice of `text`, so a stored
        // key outlives the borrow of the line it was read from.
        let key_name = match canonical_key(key) {
            Some(name) => name,
            None => return err(line, format!("unknown key `{key}`")),
        };
        let stored = if key_name == "budget_s" {
            let number: u64 = value.parse().map_err(|_| SyntaxError {
                line,
                message: format!("budget_s is not a non-negative integer: {value}"),
            })?;
            if number == 0 || number > MAX_BUDGET_S {
                return err(
                    line,
                    format!("budget_s {number} is outside 1..={MAX_BUDGET_S}"),
                );
            }
            number.to_string()
        } else {
            if value.is_empty() {
                return err(line, format!("key `{key}` has no value"));
            }
            string_value(value, line)?
        };
        fields.push((key_name, stored));
    }

    if !fields.is_empty() {
        finish_row(&mut fields, &mut rows, row_line)?;
    }
    Ok(rows)
}

/// Collect one row from the keys seen since its header. Every schema key is required, so an
/// incomplete row is a refusal rather than a row of defaults.
fn finish_row(
    fields: &mut Vec<(&'static str, String)>,
    rows: &mut Vec<Row>,
    row_line: usize,
) -> Result<(), SyntaxError> {
    let take = |name: &str| {
        fields
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.clone())
    };
    let missing = |name: &str| err::<()>(row_line, format!("row is missing required key `{name}`"));
    let id = match take("id") {
        Some(value) => value,
        None => return missing("id"),
    };
    let kind = match take("kind") {
        Some(value) => value,
        None => return missing("kind"),
    };
    let path = match take("path") {
        Some(value) => value,
        None => return missing("path"),
    };
    let sha256 = match take("sha256") {
        Some(value) => digest_value(&value, row_line)?,
        None => return missing("sha256"),
    };
    let gate = match take("gate") {
        Some(value) => value,
        None => return missing("gate"),
    };
    let control = match take("control") {
        Some(value) => value,
        None => return missing("control"),
    };
    let budget_s = match take("budget_s") {
        Some(value) => value.parse().map_err(|_| SyntaxError {
            line: row_line,
            message: "budget_s is not an integer".to_string(),
        })?,
        None => return missing("budget_s"),
    };
    if !KINDS.contains(&kind.as_str()) {
        return err(
            row_line,
            format!("kind `{kind}` is not one of {}", KINDS.join(" | ")),
        );
    }
    if id.is_empty() {
        return err(row_line, "id is empty");
    }
    rows.push(Row {
        id,
        kind,
        path,
        sha256,
        gate,
        control,
        budget_s,
    });
    fields.clear();
    Ok(())
}

// -- hashing ---------------------------------------------------------------------------

/// FIPS 180-4 round constants.
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// FIPS 180-4 initial hash value.
const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// The sha256 of `bytes`, lowercase hex. Implemented here rather than pulled from a crate because
/// this tool is a trust boundary over the repository's own gates: its own executable is the
/// smallest dependency set that can carry a digest.
fn sha256(bytes: &[u8]) -> String {
    let mut state = IV;
    for block in bytes.chunks_exact(64) {
        compress(&mut state, block);
    }

    // Final block(s): the 0x80 byte, zeros, and the 64-bit big-endian bit length. A remainder of
    // 56 bytes or more does not leave room for the length, so it costs a second block.
    let remainder = bytes.len() % 64;
    let mut block = [0u8; 64];
    block[..remainder].copy_from_slice(&bytes[bytes.len() - remainder..]);
    block[remainder] = 0x80;
    if remainder >= 56 {
        compress(&mut state, &block);
        block = [0u8; 64];
    }
    let bits = (bytes.len() as u64).wrapping_mul(8);
    block[56..].copy_from_slice(&bits.to_be_bytes());
    compress(&mut state, &block);

    state.iter().map(|word| format!("{word:08x}")).collect()
}

/// One 64-byte block: expand to a 64-word schedule, then 64 rounds of the compression function.
fn compress(state: &mut [u32; 8], block: &[u8]) {
    debug_assert_eq!(block.len(), 64);
    let mut schedule = [0u32; 64];
    for (word, slot) in schedule.iter_mut().take(16).enumerate() {
        let base = word * 4;
        *slot = u32::from_be_bytes([
            block[base],
            block[base + 1],
            block[base + 2],
            block[base + 3],
        ]);
    }
    for word in 16..64 {
        let s0 = schedule[word - 15].rotate_right(7)
            ^ schedule[word - 15].rotate_right(18)
            ^ (schedule[word - 15] >> 3);
        let s1 = schedule[word - 2].rotate_right(17)
            ^ schedule[word - 2].rotate_right(19)
            ^ (schedule[word - 2] >> 10);
        schedule[word] = schedule[word - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[word - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for word in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[word])
            .wrapping_add(schedule[word]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *slot = slot.wrapping_add(value);
    }
}

/// Whether a path component is skipped when a directory digest is built.
fn excluded(name: &str) -> bool {
    name.starts_with('.') || name == "target"
}

/// Collect every file under `dir` as a `(relative path, absolute path)` pair, sorted by the
/// relative path. Excludes dot components and `target`. Iterative walk with an explicit stack, so
/// the recursion depth is bounded by the pushed list rather than the call stack.
fn walk(dir: &Path, prefix: &Path, out: &mut Vec<(PathBuf, PathBuf)>) -> io::Result<()> {
    let mut pending: Vec<(PathBuf, PathBuf)> = vec![(prefix.to_path_buf(), dir.to_path_buf())];
    while let Some((relative, absolute)) = pending.pop() {
        for entry in fs::read_dir(&absolute)? {
            let entry = entry?;
            let name = entry.file_name();
            if excluded(&name.to_string_lossy()) {
                continue;
            }
            let file_type = entry.file_type()?;
            let child_relative = relative.join(&name);
            let child_absolute = absolute.join(&name);
            if file_type.is_dir() {
                pending.push((child_relative, child_absolute));
            } else if file_type.is_file() {
                out.push((child_relative, child_absolute));
            }
        }
    }
    out.sort();
    Ok(())
}

/// The identity of `path` under the rule stated at the top of this file: a file hashes its bytes, a
/// directory hashes the sorted `<hex>  <repo-relative path>` listing of its contents.
pub fn sha256_path(path: &Path, relative_to: &Path) -> io::Result<String> {
    let meta = fs::metadata(path)?;
    if meta.is_file() {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        return Ok(sha256(&bytes));
    }
    if !meta.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} is neither a file nor a directory", path.display()),
        ));
    }

    let mut files = Vec::new();
    walk(path, Path::new(""), &mut files)?;
    let mut listing = String::new();
    for (relative, absolute) in &files {
        let mut bytes = Vec::new();
        File::open(absolute)?.read_to_end(&mut bytes)?;
        listing.push_str(&sha256(&bytes));
        listing.push_str("  ");
        listing.push_str(&strip_prefix(relative, relative_to).to_string_lossy());
        listing.push('\n');
    }
    Ok(sha256(listing.as_bytes()))
}

/// `path` relative to `base` when it is below it, and unchanged otherwise. The listing records
/// repository-relative names, so a row registered from a different working directory still digests
/// the same.
fn strip_prefix<'a>(path: &'a Path, base: &Path) -> &'a Path {
    path.strip_prefix(base).unwrap_or(path)
}

// -- process, clock, lock --------------------------------------------------------------

// `Outcome` and `run_budgeted` are the budgeted-execution primitive. Nothing in this wave runs a
// gate — `docs/plan.md` E1 W4 owns the aggregate that does — so in a build without the test
// configuration they are unused, and the `allow` says exactly that rather than leaving a warning
// or an invented caller. The controls below are the reason they exist and the reason they work.

/// What a budgeted child did. `timed_out` is the load-bearing field: a child killed at the budget
/// did not report a result, and reporting its exit code as a verdict would be the quiet success
/// `AGENTS.md` §10 records as a failure.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    pub code: Option<i32>,
    pub timed_out: bool,
    pub elapsed_ms: u128,
}

#[allow(dead_code)]
impl Outcome {
    /// A verdict in the terms a gate report uses: refused unless it exited 0 inside its budget.
    pub fn accepted(&self) -> bool {
        !self.timed_out && self.code == Some(0)
    }
}

/// Run `command` through `/bin/sh -c` in `cwd`, killing it once `budget_s` has elapsed. The child's
/// stdio is inherited, so its markers reach the terminal directly and no pipe can fill and deadlock
/// the wait. Returns as soon as the child exits, or shortly after the budget expires.
///
/// Not sanitising the command string is deliberate and is the trust boundary stated at the top of
/// this file.
#[allow(dead_code)]
pub fn run_budgeted(command: &str, cwd: &Path, budget_s: u64) -> io::Result<Outcome> {
    let start = Instant::now();
    let mut child = Command::new("/bin/sh")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;
    let budget = Duration::from_secs(budget_s);
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Outcome {
                code: status.code(),
                timed_out: false,
                elapsed_ms: start.elapsed().as_millis(),
            });
        }
        if start.elapsed() >= budget {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(Outcome {
                code: None,
                timed_out: true,
                elapsed_ms: start.elapsed().as_millis(),
            });
        }
        std::thread::sleep(Duration::from_millis(POLL_MS));
    }
}

/// An exclusive lock file, released when it is dropped.
///
/// Acquired by creating the path with `O_EXCL`, so the winner is decided by the filesystem and two
/// callers cannot both hold it. A lock left behind by a killed process is *not* stolen: a lock that
/// can be broken without evidence is not a lock. The wait is bounded by `budget_s`, after which the
/// caller is told the lock is held rather than being told it is free.
#[derive(Debug)]
pub struct Lock {
    path: PathBuf,
}

impl Lock {
    /// Take `path`, waiting up to `budget_s` for a holder to release it.
    pub fn acquire(path: &Path, budget_s: u64) -> io::Result<Lock> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let start = Instant::now();
        loop {
            match File::options().write(true).create_new(true).open(path) {
                Ok(mut file) => {
                    use std::io::Write;
                    writeln!(file, "{}", std::process::id())?;
                    return Ok(Lock {
                        path: path.to_path_buf(),
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    if start.elapsed() >= Duration::from_secs(budget_s) {
                        return Err(io::Error::new(
                            io::ErrorKind::WouldBlock,
                            format!("lock {} still held after {budget_s}s", path.display()),
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(LOCK_POLL_MS));
                }
                Err(error) => return Err(error),
            }
        }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Read a file as text. A path that does not resolve is an error the caller reports, never a
/// silently skipped row.
pub fn read_text(path: &Path) -> io::Result<String> {
    fs::read_to_string(path)
}

/// Load the registry at `path`: read it, then parse it. The parse result is a syntax verdict, not
/// an admission; `core::validate` and `core::admit` decide.
pub fn load_manifest(path: &Path) -> Result<Vec<Row>, String> {
    let text = read_text(path).map_err(|error| format!("{}: {error}", path.display()))?;
    parse(&text).map_err(|error| format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod controls {
    use super::*;

    /// The hasher must agree with a known digest of the empty string and of `abc`, or every
    /// recorded identity in the registry is wrong in a way nothing else would notice.
    #[test]
    fn sha256_matches_fips_vectors() {
        assert_eq!(
            sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // 56 bytes: the padding case that needs a second block.
        assert_eq!(
            sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// A field the schema does not admit is refused at the syntax layer, not defaulted: a digest
    /// that is not 64 hex digits cannot be compared, and a `kind` outside the enum is not a kind.
    #[test]
    fn malformed_registry_is_refused() {
        let good = "0".repeat(64);
        let bad_digest = "[[artifact]]\nid = \"a\"\nkind = \"rust\"\npath = \"a.rs\"\n\
                        sha256 = \"nothex\"\ngate = \"cargo test\"\ncontrol = \"\"\nbudget_s = 540\n";
        let error = parse(bad_digest).expect_err("a non-hex digest must be refused");
        assert!(error.message.contains("64 lowercase hex digits"), "{error}");

        let bad_kind = format!(
            "[[artifact]]\nid = \"a\"\nkind = \"sigil\"\npath = \"a.rs\"\nsha256 = \"{good}\"\n\
             gate = \"cargo test\"\ncontrol = \"\"\nbudget_s = 540\n"
        );
        let error = parse(&bad_kind).expect_err("an unknown kind must be refused");
        assert!(error.message.contains("kind `sigil`"), "{error}");

        let over_budget = format!(
            "[[artifact]]\nid = \"a\"\nkind = \"rust\"\npath = \"a.rs\"\nsha256 = \"{good}\"\n\
             gate = \"cargo test\"\ncontrol = \"\"\nbudget_s = 900\n"
        );
        let error = parse(&over_budget).expect_err("a budget past the ceiling must be refused");
        assert!(error.message.contains("outside 1..=600"), "{error}");

        let missing = "[[artifact]]\nid = \"a\"\nkind = \"rust\"\npath = \"a.rs\"\n";
        let error = parse(missing).expect_err("an incomplete row must be refused");
        assert!(
            error.message.contains("missing required key `sha256`"),
            "{error}"
        );
    }

    /// A child that outlives its declared budget is killed and reported as `timed_out`, never as an
    /// exit code: a gate killed at the ceiling produced no result, and reading its signal death as
    /// a verdict is the quiet success the failure register records.
    #[test]
    fn budget_kills_a_child_that_outlasts_it() {
        let here = Path::new(".");
        let overrun = run_budgeted("sleep 30", here, 1).expect("spawn");
        assert!(overrun.timed_out, "{overrun:?}");
        assert!(!overrun.accepted());
        assert!(
            overrun.elapsed_ms < 10_000,
            "the budget must bound the wait: {}ms",
            overrun.elapsed_ms
        );

        let inside = run_budgeted("exit 0", here, 60).expect("spawn");
        assert_eq!(inside.code, Some(0), "{inside:?}");
        assert!(inside.accepted());

        let failed = run_budgeted("exit 3", here, 60).expect("spawn");
        assert_eq!(failed.code, Some(3), "{failed:?}");
        assert!(!failed.accepted());
    }

    /// A lock is exclusive while it is held, and is released when the holder drops it. The second
    /// acquire is bounded, so a lock that is never released is reported rather than waited on
    /// forever.
    #[test]
    fn lock_is_exclusive_until_dropped() {
        let path = std::env::temp_dir().join(format!("gate-lock-{}.lock", std::process::id()));
        let held = Lock::acquire(&path, 1).expect("first acquire");
        assert!(path.is_file(), "the lock file exists while it is held");

        let blocked = Lock::acquire(&path, 1).expect_err("a held lock is not re-entered");
        assert_eq!(blocked.kind(), io::ErrorKind::WouldBlock, "{blocked}");

        drop(held);
        let second = Lock::acquire(&path, 1).expect("released lock is re-acquirable");
        drop(second);
    }
}
