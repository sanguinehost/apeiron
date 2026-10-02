//! Impure boundary for `tools/gate`: filesystem, hashing, subprocess, wall clock and locks.
//!
//! `core.rs` is the pure admission decision. This file supplies what a pure body cannot have, and
//! nothing else: it reads a file, hashes it, runs a command under a declared wall-clock budget and
//! takes a lock file. It formats what it observed and decides nothing about admission; every
//! verdict comes from `core::validate` or `core::admit`. The split mirrors
//! `tools/dispatch/{core.rs,main.rs}`.
//!
//! **Trust boundary.** The command string handed to [`run_captured`] is executed by `/bin/sh -c`
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
//! **A recorded run.** [`run_captured`] returns a [`RunRecord`]: the child's exit status, whether it
//! was killed at the budget, its elapsed time, and the sha256 of what it printed on each stream.
//! That record is what a receipt names, so a run's result and its output identity are one value
//! rather than a claim about a terminal scrollback. [`run_captured_with_tail`] is the same run and
//! also returns a bounded excerpt of what the child printed, which is what a refusal quotes: the
//! record binds the whole output, the tail is what a reader can read.
//!
//! **Complexity.** [`sha256_path`] on a file is O(bytes) time and O(1) space beyond the 64-byte
//! padding block: one pass, no buffering of the payload. On a directory it is O(files + bytes) time
//! and O(files) space in the sorted path list; the sort is over the collected names, and the
//! collection is one directory walk that terminates because the walk is over a finite tree.
//! [`run_captured`] is O(`COPY_CHUNK`) space whatever the child prints, and its wall time is
//! bounded by `budget_s + POLL_MS + output bytes`: the child is polled every [`POLL_MS`] and killed
//! on expiry, then its two output files are relayed once. [`relay`] is O(bytes) time and
//! O(`COPY_CHUNK`) space. [`run_captured_with_tail`] adds O(`tail_limit`) space and O(bytes kept)
//! work over the same two passes; [`Tail::push`] drains from the front only when a chunk overflows
//! the limit, so a child that stays under it costs one `extend_from_slice` per chunk.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Poll interval for a budgeted child. The overshoot past the declared budget is bounded by this.
const POLL_MS: u64 = 25;

/// Bytes moved per iteration when relaying a child's output. Bounds the relay buffer.
const COPY_CHUNK: usize = 8192;

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

/// A streaming sha256: the eight-word state, one 64-byte block of carry-over, and the 64-bit bit
/// count. Space is O(1) in the message length, so a gate that prints a gigabyte is digested with
/// one block of carry-over rather than by holding its output.
///
/// Time O(bytes) per `update` and O(1) per `finish`; space O(1).
pub struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    held: usize,
    bits: u64,
}

impl Sha256 {
    /// The initial hash value of FIPS 180-4 §5.3.3, with an empty block and a zero bit count.
    pub fn new() -> Sha256 {
        Sha256 {
            state: IV,
            block: [0u8; 64],
            held: 0,
            bits: 0,
        }
    }

    /// Absorb `bytes`. A full block is compressed in place; the remainder is carried into the next
    /// call, so a message split anywhere digests to the same value as the whole message.
    pub fn update(&mut self, bytes: &[u8]) {
        self.bits = self.bits.wrapping_add((bytes.len() as u64).wrapping_mul(8));
        let mut rest = bytes;
        if self.held > 0 {
            let want = 64 - self.held;
            let take = want.min(rest.len());
            self.block[self.held..self.held + take].copy_from_slice(&rest[..take]);
            self.held += take;
            rest = &rest[take..];
            if self.held == 64 {
                compress(&mut self.state, &self.block);
                self.held = 0;
                // The block is spent: zero it, because `finish` writes whole bytes of padding and a
                // tail copied over stale bytes would digest a message the caller never sent.
                self.block = [0u8; 64];
            } else {
                // The carry-over is short of a block and `rest` is exhausted, so the bytes just
                // written are still the tail. Returning is what keeps them.
                return;
            }
        }
        let mut chunks = rest.chunks_exact(64);
        for chunk in &mut chunks {
            compress(&mut self.state, chunk);
        }
        let tail = chunks.remainder();
        self.block[..tail.len()].copy_from_slice(tail);
        self.held = tail.len();
    }

    /// The digest, lowercase hex. Pads with `0x80`, zeros and the big-endian bit count; a carry-over
    /// of 56 bytes or more leaves no room for the length and costs a second compression.
    pub fn finish(mut self) -> String {
        let held = self.held;
        self.block[held] = 0x80;
        if held >= 56 {
            compress(&mut self.state, &self.block);
            self.block = [0u8; 64];
        }
        self.block[56..].copy_from_slice(&self.bits.to_be_bytes());
        compress(&mut self.state, &self.block);
        self.state
            .iter()
            .map(|word| format!("{word:08x}"))
            .collect()
    }
}

/// The sha256 of `bytes`, lowercase hex. Implemented here rather than pulled from a crate because
/// this tool is a trust boundary over the repository's own gates: its own executable is the
/// smallest dependency set that can carry a digest.
fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finish()
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

// `RunRecord`, `RunRefusal` and `run_captured` are the budgeted-execution primitive. `gate run`
// (`main.rs`) executes a row's `gate` column through `run_captured_with_tail`, which is this same
// body plus the refusal tail; `docs/plan.md` E1 W4 owns the aggregate `gate docs check --controls`
// over those runs. In a build without the test configuration the tail-free entry point is unused, so
// the `allow` says exactly that rather than leaving a warning or an invented caller. The controls
// below are the reason they exist and the reason they work.

/// Why a run is refused, as the adapter sees it. Not a core `Refusal`: admission is
/// `core.rs`'s decision, and these are the observations a pure decision cannot make for itself.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunRefusal {
    /// The child could not be started, or its output could not be read back. No verdict exists.
    Unrunnable { command: String, message: String },
    /// The child outlived its declared budget and was killed at it. It reported nothing, so reading
    /// its absence of an exit status as success is the quiet success `AGENTS.md` §10 records.
    BudgetExceeded {
        command: String,
        budget_s: u64,
        elapsed_ms: u128,
    },
}

#[allow(dead_code)]
impl std::fmt::Display for RunRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunRefusal::Unrunnable { command, message } => {
                write!(f, "cannot run `{command}`: {message}")
            }
            RunRefusal::BudgetExceeded {
                command,
                budget_s,
                elapsed_ms,
            } => write!(
                f,
                "`{command}` was killed at its budget of {budget_s}s after {elapsed_ms}ms; it reported no exit status"
            ),
        }
    }
}

/// What a budgeted child did, and the identity of what it printed. `timed_out` and the two digests
/// are the load-bearing fields: a child killed at the budget produced no result, and its output is
/// what a receipt must name.
///
/// A non-zero exit is not a [`RunRefusal`]: it is the child's own verdict, carried by `code`, and
/// read with [`RunRecord::accepted`].
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunRecord {
    /// The exact string handed to `/bin/sh -c`, recorded so a receipt names what ran.
    pub command: String,
    /// The working directory it ran in, as given.
    pub cwd: String,
    /// The budget it was killed at, if any.
    pub budget_s: u64,
    /// The child's exit status, or `None` if it was killed at the budget or died on a signal.
    pub code: Option<i32>,
    pub timed_out: bool,
    pub elapsed_ms: u128,
    pub stdout_sha256: String,
    pub stderr_sha256: String,
}

#[allow(dead_code)]
impl RunRecord {
    /// A verdict in the terms a gate report uses: accepted unless it exited 0 inside its budget.
    pub fn accepted(&self) -> bool {
        !self.timed_out && self.code == Some(0)
    }

    /// The refusal this run carries, or `None` when it reported an exit status.
    pub fn refusal(&self) -> Option<RunRefusal> {
        if self.timed_out {
            return Some(RunRefusal::BudgetExceeded {
                command: self.command.clone(),
                budget_s: self.budget_s,
                elapsed_ms: self.elapsed_ms,
            });
        }
        None
    }
}

/// Distinguishes two runs in one process, so their scratch directories cannot collide.
static RUN_SEQ: AtomicU64 = AtomicU64::new(0);

/// A per-run scratch directory, removed when it goes out of scope. Every path out of
/// [`run_captured`] therefore cleans up, including the refusal paths that return before a child was
/// ever started: a gate runner that leaves a directory behind on every failed run fills a
/// temporary filesystem with the evidence of its own failures.
#[derive(Debug)]
struct RunDir(PathBuf);

impl Drop for RunDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A scratch directory named from the process, the clock and a process-local counter. Callers keep
/// it alive with a [`RunDir`] binding.
fn run_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let seq = RUN_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("gate-run-{}-{nanos}-{seq}", std::process::id()))
}

/// Stream `path` into `out` and return the sha256 of its bytes. One pass, [`COPY_CHUNK`] bytes at
/// a time, so the space is O(`COPY_CHUNK`) whatever the file holds and the child never waits on an
/// unread pipe.
fn relay(path: &Path, out: &mut dyn Write) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; COPY_CHUNK];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        out.write_all(&buffer[..read])?;
    }
    out.flush()?;
    Ok(hasher.finish())
}

/// The last `limit` bytes written to it, and nothing else. A child may print without bound and this
/// tool's memory may not, so a refusal's excerpt is bounded by the same constant as the relay's
/// buffer rather than by the child's output.
#[derive(Debug, Default)]
struct Tail {
    bytes: Vec<u8>,
    limit: usize,
}

impl Tail {
    fn new(limit: usize) -> Tail {
        Tail {
            bytes: Vec::new(),
            limit,
        }
    }

    /// Push `chunk`, dropping from the front until at most `limit` bytes remain. Amortised O(bytes
    /// kept) per call: a chunk that fits appends, and only a full chunk costs the drain.
    fn push(&mut self, chunk: &[u8]) {
        self.bytes.extend_from_slice(chunk);
        if self.bytes.len() > self.limit {
            let drop = self.bytes.len() - self.limit;
            self.bytes.drain(..drop);
        }
    }
}

impl Write for Tail {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.push(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Forwards every byte to the caller's stream and keeps the last `limit` of them. The relayed bytes
/// are the same bytes that were digested, so the tail and `RunRecord::stdout_sha256` name one
/// output rather than two.
struct Tee<'a> {
    out: &'a mut dyn Write,
    tail: &'a mut Tail,
}

impl Write for Tee<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.out.write_all(buf)?;
        self.tail.push(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

/// Run `command` through `/bin/sh -c` in `cwd`, killing it once `budget_s` has elapsed, and record
/// its exit status together with the sha256 of what it printed on each stream.
///
/// The child's stdout and stderr go to files under a scratch directory rather than to pipes. A pipe
/// nobody reads fills and deadlocks the wait, and a child's output is unbounded while this tool's
/// memory is not; a file blocks neither. The two files are relayed to the parent's own streams and
/// digested in one pass after the child is reaped, so its markers still reach the terminal and the
/// record's digests are the bytes that reached it.
///
/// Not sanitising the command string is deliberate and is the trust boundary stated at the top of
/// this file.
///
/// The digests bind what the direct child wrote before it was reaped. An orphaned grandchild that
/// inherited the descriptors can still append afterwards; nothing here can see that, and it is
/// recorded here rather than claimed closed.
///
/// The tail-free entry point, for a caller that needs only the record.
#[allow(dead_code)]
pub fn run_captured(command: &str, cwd: &Path, budget_s: u64) -> Result<RunRecord, RunRefusal> {
    run_captured_with_tail(command, cwd, budget_s, 0).map(|(record, _)| record)
}

/// [`run_captured`], plus the last `tail_limit` bytes the child wrote, as one string: stdout's tail
/// then stderr's, in the order they were relayed.
///
/// The tail is what a refusal quotes, and it is bounded by `tail_limit` rather than by the child's
/// output: a caller that printed a child's whole gigabyte would report the failure with the failure.
/// A `tail_limit` of 0 collects nothing.
///
/// Time and space are `run_captured`'s plus O(`tail_limit`); the tail is drained from the front on
/// each full chunk, so the work is bounded by the bytes kept, not by the bytes printed.
pub fn run_captured_with_tail(
    command: &str,
    cwd: &Path,
    budget_s: u64,
    tail_limit: usize,
) -> Result<(RunRecord, String), RunRefusal> {
    let unrunnable = |message: String| RunRefusal::Unrunnable {
        command: command.to_string(),
        message,
    };

    let dir = run_dir();
    fs::create_dir_all(&dir).map_err(|error| unrunnable(format!("{}: {error}", dir.display())))?;
    // Kept alive to the end of the function so every return path removes the scratch directory.
    let _scratch = RunDir(dir.clone());
    let stdout_path = dir.join("stdout");
    let stderr_path = dir.join("stderr");
    let stdout_file = File::create(&stdout_path)
        .map_err(|error| unrunnable(format!("{}: {error}", stdout_path.display())))?;
    let stderr_file = File::create(&stderr_path)
        .map_err(|error| unrunnable(format!("{}: {error}", stderr_path.display())))?;

    let start = Instant::now();
    let mut child = Command::new("/bin/sh")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
        .map_err(|error| unrunnable(format!("in {}: {error}", cwd.display())))?;
    let budget = Duration::from_secs(budget_s);
    let (code, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(status)) => break (status.code(), false),
            Ok(None) => {}
            Err(error) => return Err(unrunnable(error.to_string())),
        }
        if start.elapsed() >= budget {
            let _ = child.kill();
            let _ = child.wait();
            break (None, true);
        }
        std::thread::sleep(Duration::from_millis(POLL_MS));
    };
    let elapsed_ms = start.elapsed().as_millis();

    let mut stdout_tail = Tail::new(tail_limit);
    let mut stderr_tail = Tail::new(tail_limit);
    let (stdout_sha256, stderr_sha256) = {
        let mut out = Tee {
            out: &mut std::io::stdout(),
            tail: &mut stdout_tail,
        };
        let stdout_sha256 = relay(&stdout_path, &mut out).map_err(|e| unrunnable(e.to_string()))?;
        let mut err = Tee {
            out: &mut std::io::stderr(),
            tail: &mut stderr_tail,
        };
        let stderr_sha256 = relay(&stderr_path, &mut err).map_err(|e| unrunnable(e.to_string()))?;
        (stdout_sha256, stderr_sha256)
    };
    let mut tail = String::from_utf8_lossy(&stdout_tail.bytes).into_owned();
    tail.push_str(&String::from_utf8_lossy(&stderr_tail.bytes));

    Ok((
        RunRecord {
            command: command.to_string(),
            cwd: cwd.to_string_lossy().into_owned(),
            budget_s,
            code,
            timed_out,
            elapsed_ms,
            stdout_sha256,
            stderr_sha256,
        },
        tail,
    ))
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
pub(crate) mod controls {
    use super::*;
    use std::sync::Mutex;

    /// Serialises every control that counts scratch directories — here and in `main.rs` — so a run
    /// another control has in flight cannot be counted as a leak. The census in [`scratch_dirs`] is
    /// process-wide and its prefix carries the process id, so a `run` control in `main.rs` would
    /// otherwise show up in this module's before/after snapshot and fail the leak check it exists
    /// to make. Each is an IO-bound subprocess test, so serialising them costs the suite about the
    /// wall time of one `sleep`.
    pub(crate) static SCRATCH: Mutex<()> = Mutex::new(());

    /// Hold the scratch census for the duration of a control that creates scratch directories.
    pub(crate) fn serialise_scratch() -> std::sync::MutexGuard<'static, ()> {
        SCRATCH.lock().unwrap_or_else(|error| error.into_inner())
    }

    /// The scratch directories this process has created and not yet removed.
    fn scratch_dirs() -> Vec<PathBuf> {
        let prefix = format!("gate-run-{}-", std::process::id());
        let Ok(entries) = fs::read_dir(std::env::temp_dir()) else {
            return Vec::new();
        };
        entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with(&prefix))
            })
            .collect()
    }

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

    /// The streaming hasher must agree with the one-shot one at every split, because a gate's
    /// output is relayed in `COPY_CHUNK` pieces and digested as it goes. A boundary error here
    /// would record a digest no other implementation reproduces.
    #[test]
    fn streamed_digest_matches_the_one_shot_digest() {
        let message: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        let expected = sha256(&message);
        for split in [0, 1, 55, 56, 63, 64, 65, 127, 128, 500, 999, 1000] {
            let mut hasher = Sha256::new();
            hasher.update(&message[..split]);
            hasher.update(&message[split..]);
            assert_eq!(hasher.finish(), expected, "split at {split}");
        }
        // The empty stream, which a child that printed nothing produces.
        assert_eq!(Sha256::new().finish(), sha256(b""));
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

    /// A child that outlives its declared budget is killed and reported as `timed_out` with an
    /// attributable refusal, never as an exit code: a gate killed at the ceiling produced no
    /// result, and reading its signal death as a verdict is the quiet success the failure register
    /// records. The refusal names the command, the budget and the elapsed time, and the positive
    /// restores it inside its budget.
    #[test]
    fn over_budget_run_is_refused() {
        let _serial = SCRATCH.lock().unwrap_or_else(|error| error.into_inner());
        let before = scratch_dirs();
        let here = Path::new(".");
        let overrun = run_captured("sleep 30", here, 1).expect("spawn");
        assert!(overrun.timed_out, "{overrun:?}");
        assert!(
            !overrun.accepted(),
            "a killed child reported nothing to accept"
        );

        let refusal = overrun.refusal().expect("a killed child is a refusal");
        assert_eq!(
            refusal,
            RunRefusal::BudgetExceeded {
                command: "sleep 30".into(),
                budget_s: 1,
                elapsed_ms: overrun.elapsed_ms,
            }
        );
        let text = refusal.to_string();
        assert!(text.contains("sleep 30"), "{text}");
        assert!(text.contains("budget of 1s"), "{text}");
        assert!(
            overrun.elapsed_ms < 10_000,
            "the budget must bound the wait: {}ms",
            overrun.elapsed_ms
        );

        // Restore: the same runner, inside the budget, is accepted and carries no refusal.
        let inside = run_captured("exit 0", here, 60).expect("spawn");
        assert!(inside.accepted(), "{inside:?}");
        assert_eq!(inside.refusal(), None, "{inside:?}");
        assert_eq!(inside.code, Some(0), "{inside:?}");
        assert_eq!(scratch_dirs(), before, "a run leaves no scratch behind");
    }

    /// A run records the exit status *and* the identity of what it printed, and a changed byte
    /// changes the recorded digest. Without the digest a record names a status but not the output
    /// it came from, and two runs that differ only in output would be indistinguishable.
    #[test]
    fn run_records_exit_status_and_output_digest() {
        let _serial = SCRATCH.lock().unwrap_or_else(|error| error.into_inner());
        let before = scratch_dirs();
        let here = Path::new(".");
        let quiet = run_captured("printf 'green'", here, 60).expect("spawn");
        assert_eq!(quiet.code, Some(0), "{quiet:?}");
        assert_eq!(quiet.stdout_sha256, sha256(b"green"));
        assert_eq!(quiet.stderr_sha256, sha256(b""));

        // Same status, one byte of output changed: the digest must move with it.
        let red = run_captured("printf 'red'; exit 1", here, 60).expect("spawn");
        assert_eq!(red.code, Some(1), "{red:?}");
        assert_eq!(red.stdout_sha256, sha256(b"red"));
        assert_ne!(red.stdout_sha256, quiet.stdout_sha256);

        // The two streams are digested apart, so output on stderr is not conflated with stdout.
        let split = run_captured("printf out; printf err >&2", here, 60).expect("spawn");
        assert_eq!(split.stdout_sha256, sha256(b"out"), "{split:?}");
        assert_eq!(split.stderr_sha256, sha256(b"err"), "{split:?}");
        assert_eq!(scratch_dirs(), before, "a run leaves no scratch behind");
    }

    /// The relay digests output larger than its own buffer, in `COPY_CHUNK` pieces, and hands back
    /// the same bytes it digested. A gate printing more than 8 KiB crosses the read boundary here;
    /// a relay that digested only the first chunk would record an identity no other reader of that
    /// file reproduces.
    #[test]
    fn relay_digests_output_larger_than_its_buffer() {
        let _serial = SCRATCH.lock().unwrap_or_else(|error| error.into_inner());
        let before = scratch_dirs();
        let dir = run_dir();
        fs::create_dir_all(&dir).expect("scratch dir");
        let _scratch = RunDir(dir.clone());
        let path = dir.join("stdout");
        // 3 full `COPY_CHUNK` reads and a 300-byte remainder, spanning four block boundaries.
        let bytes = vec![b'a'; COPY_CHUNK * 3 + 300];
        fs::write(&path, &bytes).expect("scratch output");

        let mut sink: Vec<u8> = Vec::new();
        let hex = relay(&path, &mut sink).expect("relay");
        assert_eq!(hex, sha256(&bytes));
        assert_eq!(sink, bytes);
        drop(_scratch);
        assert_eq!(scratch_dirs(), before, "the relay's scratch is removed");
    }

    /// A command that cannot be started is refused, not reported as a result: an unrun gate is not a
    /// passing gate. `sh` itself resolves on this host, so the failing part is the working
    /// directory, which is the input the caller controls. The refusal path must also leave no
    /// scratch directory behind: the one population that fails most is the one whose leftovers
    /// accumulate.
    #[test]
    fn unrunnable_command_is_refused() {
        let _serial = SCRATCH.lock().unwrap_or_else(|error| error.into_inner());
        let before = scratch_dirs();
        let missing_cwd = Path::new("/nonexistent-gate-cwd");
        let refusal = run_captured("exit 0", missing_cwd, 1).expect_err("a bad cwd cannot spawn");
        assert!(
            matches!(refusal, RunRefusal::Unrunnable { .. }),
            "{refusal:?}"
        );
        let text = refusal.to_string();
        assert!(text.contains("exit 0"), "{text}");
        assert!(text.contains("/nonexistent-gate-cwd"), "{text}");
        assert_eq!(
            scratch_dirs(),
            before,
            "a refused run must leave no scratch directory"
        );
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
