//! Exit codes and the machine-readable verdict block, shared by every subcommand.
//!
//! One writer builds each `key=value` block, in a fixed field order with every string quoted and
//! escaped, so a receipt parses it without a JSON reader. Nothing here decides anything: the codes
//! are the epic's (`docs/epics/gate-driver.md:52-54`) and the block is the report format.

/// Exit codes `gate run` returns. `docs/epics/gate-driver.md:52-54`: "The driver exits non-zero on
/// any stage failure with the stage and the tool named"; "a mutation that leaves the proof green is
/// exit 1 (the check could not fail)"; "Budget expiry is exit 124 with elapsed time and the
/// incomplete stage". `2` is the environment verdict `docs check` already returns — the registry
/// cannot be read or parsed, or the child could not be started at all — and is not a refusal of the
/// row's gate.
pub(crate) const EXIT_ACCEPTED: u8 = 0;
pub(crate) const EXIT_REFUSED: u8 = 1;
pub(crate) const EXIT_ENVIRONMENT: u8 = 2;
pub(crate) const EXIT_BUDGET: u8 = 124;

/// Exit `gate docs check --controls` returns when it could not finish the selection: a row was
/// blocked, or the invocation budget ran out before a row was reached. Distinct from
/// [`EXIT_REFUSED`] because the two need different readers — a refusal says a control ran and did
/// not fire, which is a defect to fix; `3` says the aggregate did not get to that row, which is a
/// batch that is not yet an answer. Both are non-zero, and neither is ever reported as success.
pub(crate) const EXIT_INCOMPLETE: u8 = 3;
/// Bytes of the child's own output kept for a refusal tail. The streams are unbounded; the excerpt
/// is not, and a refusal that grows with its child's output is not a diagnostic.
pub(crate) const TAIL_BYTES: usize = 4096;
/// One `key=value` line per field, in insertion order, every string value quoted and escaped. Fixed
/// order and quoting are what make the block parseable by a receipt that has no JSON reader, so the
/// block is built by one writer rather than by `println!` at each decision point.
#[derive(Debug, Default)]
pub(crate) struct Verdict {
    pub(crate) fields: Vec<(String, String)>,
}

impl Verdict {
    /// A `key=value` string value: quotes, backslashes and line breaks escaped, everything else
    /// verbatim. A command string carries spaces and a refusal carries a semicolon, so an unquoted
    /// value would not survive one line.
    pub(crate) fn quoted(key: &str, value: &str) -> (String, String) {
        let mut out = String::with_capacity(value.len() + 2);
        out.push('"');
        for ch in value.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                other => out.push(other),
            }
        }
        out.push('"');
        (key.to_string(), out)
    }

    /// A `key=value` numeric field, unquoted: a number is read as a number.
    pub(crate) fn number(key: &str, value: impl std::fmt::Display) -> (String, String) {
        (key.to_string(), value.to_string())
    }

    /// The block, one line per field, each line newline-terminated.
    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        for (key, value) in &self.fields {
            out.push_str(key);
            out.push('=');
            out.push_str(value);
            out.push('\n');
        }
        out
    }
}
