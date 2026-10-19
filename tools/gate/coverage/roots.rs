//! `gate roots`: the kernel-routing gate (AGENTS.md §0 K1/K2).
//!
//! Unit: every non-test `fn` in `src/learn/**/*.rs` whose name carries a product word
//! (`dot`, `gemm`, `syrk`, `gram`, `matmul`, `matvec`). A product `fn` is **rooted** when its own
//! body calls the one limb adapter (`crate::engine::kern::limb::{gemm_q32, syrk_q32, gemv_q32,
//! rowdot_q32}`), directly or through the file's `use crate::engine::kern::limb` import. A body
//! that calls the kernel (`gemm_q8_0`) or the exact Rust body directly is not rooted: K2 names
//! `limb.rs` as the only adapter.
//!
//! `BASELINE` is the exact unrooted set at the gate's introduction. `--check` refuses a new
//! unrooted product (a second implementation) and a stale baseline row (a body that is now rooted
//! or a file that is gone), so the set only shrinks. The pre-commit hook runs `--staged` against
//! the git index, which is the commit's content; `--check` without `--staged` is the standing tree.
//!
//! Limits: the scanner is lexical, not a parser (`coverage/scan.rs`); the name rule is the staged
//! `kernel-route` rule, so an error mapper named `map_syrk` is in the baseline as a name match, not
//! a product. A wrapper that roots through a helper in another file is seen as unrooted unless
//! that helper is itself rooted; macro-generated calls are not seen.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use crate::cli::{EXIT_ACCEPTED, EXIT_ENVIRONMENT, EXIT_REFUSED};

use super::scan::scan_items;

/// The unrooted product `fn`s at introduction: 15 files, 34 fns. A file-level reference audit
/// found 8 files with no `crate::engine::kern` text at all; this body-level gate also finds the
/// products in files whose kernel mention is a sibling body, not the product's own call. Rows
/// leave only when the body is rooted through `limb.rs` and its old body is deleted or test-only;
/// a fix that leaves the set without removing the row is a `stale-baseline` refusal, so the list
/// cannot drift.
const BASELINE: &[(&str, &str)] = &[
    ("src/learn/fold/finish/plane.rs", "dot_cols"),
    ("src/learn/fold/finish/plane.rs", "row_group_dots"),
    ("src/learn/fold/fit/eigh.rs", "eigen_gram"),
    ("src/learn/fold/fit/eigh.rs", "gemm_err"),
    ("src/learn/fold/fit/eigh.rs", "gram_scale"),
    ("src/learn/fold/fit/icm.rs", "dot_q"),
    ("src/learn/fold/fit/icm.rs", "had_gram_q"),
    ("src/learn/fold/fit/icm.rs", "matmul"),
    ("src/learn/fold/fit/icm.rs", "matmul_l"),
    ("src/learn/fold/fit/steps.rs", "had_gram"),
    ("src/learn/fold/fit/steps.rs", "had_gram_grid"),
    ("src/learn/fold/fit/steps.rs", "map_syrk"),
    ("src/learn/fold/layer/resq.rs", "gram"),
    ("src/learn/fold/layer/stage.rs", "dot"),
    ("src/learn/fold/stats/basis.rs", "block_gram"),
    ("src/learn/fold/stats/basis.rs", "damped_gram"),
    ("src/learn/fold/stats/basis.rs", "map_gemm"),
    ("src/learn/fold/stats/basis.rs", "map_rowdot"),
    ("src/learn/fold/stats/basis.rs", "map_syrk"),
    ("src/learn/fold/stats/basis.rs", "rotate_gram"),
    ("src/learn/fold/stats/lw.rs", "map_syrk"),
    ("src/learn/fold/stats/parts.rs", "map_gemm"),
    ("src/learn/grad/attn.rs", "dot_round"),
    ("src/learn/grad/head.rs", "dot_round"),
    ("src/learn/gram.rs", "gram_accumulate"),
    ("src/learn/gram.rs", "gram_accumulator_clear"),
    ("src/learn/gram.rs", "gram_columns"),
    ("src/learn/gram.rs", "gram_fold_i8"),
    ("src/learn/gram.rs", "gram_fold_i8_kern"),
    ("src/learn/solve/chol.rs", "dot_shifted_col"),
    ("src/learn/solve/chol.rs", "dot_shifted_row"),
    ("src/learn/solve/gemm.rs", "gemm_nn"),
    ("src/learn/solve/gemm.rs", "gemm_nt"),
    ("src/learn/solve/syrk.rs", "wsyrk"),
];

const USAGE: &str =
    "usage: gate roots (--check | --report | --baseline) [--root DIR] [--staged]\n  \
  --check     refuse a new unrooted product fn or a stale baseline row\n  \
  --report    print every unrooted product fn and the counts\n  \
  --baseline  print the current set as BASELINE rows\n  \
  --root DIR  repository root (default .)\n  \
  --staged    read src/learn from the git index instead of the working tree";

/// Cost: time O(L) in source characters, space O(L) for the scanned sources.
pub(crate) fn roots_command(args: &[String]) -> ExitCode {
    let mut mode = None;
    let mut root = PathBuf::from(".");
    let mut staged = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--check" | "--report" | "--baseline" => mode = Some(a.clone()),
            "--root" => match it.next() {
                Some(v) => root = PathBuf::from(v),
                None => return usage("--root needs a value"),
            },
            "--staged" => staged = true,
            other => return usage(&format!("unknown argument `{other}`")),
        }
    }
    let Some(mode) = mode else {
        return usage("no mode given");
    };
    let root = match root.canonicalize() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gate roots: root {}: {e}", root.display());
            return ExitCode::from(EXIT_ENVIRONMENT);
        }
    };
    let sources = if staged {
        staged_sources(&root)
    } else {
        standing_sources(&root)
    };
    let sources = match sources {
        Ok(s) => s,
        Err(e) => {
            eprintln!("gate roots: {e}");
            return ExitCode::from(EXIT_ENVIRONMENT);
        }
    };
    let current = match unrooted(&sources) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("gate roots: {e}");
            return ExitCode::from(EXIT_ENVIRONMENT);
        }
    };
    let base = baseline();
    match mode.as_str() {
        "--report" => {
            report(&current, &base);
            ExitCode::from(EXIT_ACCEPTED)
        }
        "--baseline" => {
            for (path, f) in &current {
                println!("    (\"{path}\", \"{f}\"),");
            }
            ExitCode::from(EXIT_ACCEPTED)
        }
        _ => check(&current, &base),
    }
}

fn usage(m: &str) -> ExitCode {
    eprintln!("gate roots: {m}\n{USAGE}");
    ExitCode::from(EXIT_ENVIRONMENT)
}

fn baseline() -> BTreeSet<(String, String)> {
    BASELINE
        .iter()
        .map(|(path, f)| (path.to_string(), f.to_string()))
        .collect()
}

/// Cost: time O(S + L) over S sources and L source characters, space O(U) for U unrooted rows.
fn unrooted(sources: &[(String, String)]) -> Result<BTreeSet<(String, String)>, String> {
    let mut out = BTreeSet::new();
    for (path, src) in sources {
        let items = scan_items(src).map_err(|e| format!("{path}: {e}"))?;
        let file_limb = src.contains("crate::engine::kern::limb");
        for item in items {
            if !is_product(&item.name) || is_test_name(&item.name) {
                continue;
            }
            let Some(body) = item.body else {
                continue;
            };
            if !rooted(&body, file_limb) {
                out.insert((path.clone(), item.name));
            }
        }
    }
    Ok(out)
}

fn is_product(name: &str) -> bool {
    ["dot", "gemm", "syrk", "gram", "matmul", "matvec"]
        .iter()
        .any(|p| name.contains(p))
}

fn is_test_name(name: &str) -> bool {
    name.starts_with("test_") || name.starts_with("learn_")
}

fn rooted(body: &str, file_limb: bool) -> bool {
    body.contains("crate::engine::kern::limb")
        || (file_limb
            && ["gemm_q32", "syrk_q32", "gemv_q32", "rowdot_q32"]
                .iter()
                .any(|f| calls(body, f)))
}

/// An identifier occurrence followed by `(`: `gemm_q32(`, not `my_gemm_q32(` or `gemm_q32x`.
fn calls(body: &str, f: &str) -> bool {
    let mut from = 0;
    while let Some(k) = body[from..].find(f) {
        let i = from + k;
        let before_ok = body[..i]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
        let after_ok = body[i + f.len()..].trim_start().starts_with('(');
        if before_ok && after_ok {
            return true;
        }
        from = i + f.len();
    }
    false
}

fn report(current: &BTreeSet<(String, String)>, base: &BTreeSet<(String, String)>) {
    for (path, f) in current {
        println!("unrooted {path} {f}");
    }
    println!(
        "roots: unrooted={} baseline={} files={}",
        current.len(),
        base.len(),
        files(current).len()
    );
}

fn check(current: &BTreeSet<(String, String)>, base: &BTreeSet<(String, String)>) -> ExitCode {
    let new: Vec<_> = current.difference(base).collect();
    let stale: Vec<_> = base.difference(current).collect();
    for (path, f) in &new {
        eprintln!(
            "gate roots: unrooted-product: {path} {f}: the fn body calls no \
             crate::engine::kern::limb adapter (gemm_q32, syrk_q32, gemv_q32, rowdot_q32); a \
             product executes on the kernel through the one limb adapter (AGENTS.md §0 K1/K2)"
        );
    }
    for (path, f) in &stale {
        eprintln!(
            "gate roots: stale-baseline: {path} {f}: now rooted or absent; delete its BASELINE \
             row so the unrooted set only shrinks"
        );
    }
    let accepted = new.is_empty() && stale.is_empty();
    println!(
        "verdict={} unrooted={} baseline={} files={} new={} stale={}",
        if accepted { "accepted" } else { "refused" },
        current.len(),
        base.len(),
        files(current).len(),
        new.len(),
        stale.len()
    );
    ExitCode::from(if accepted {
        EXIT_ACCEPTED
    } else {
        EXIT_REFUSED
    })
}

fn files(set: &BTreeSet<(String, String)>) -> BTreeSet<&str> {
    set.iter().map(|(path, _)| path.as_str()).collect()
}

fn standing_sources(root: &Path) -> Result<Vec<(String, String)>, String> {
    let dir = root.join("src/learn");
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir.display()));
    }
    let mut out = Vec::new();
    walk(&dir, root, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            walk(&path, root, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let rel = path
                .strip_prefix(root)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            let src = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            out.push((rel.to_string_lossy().replace('\\', "/"), src));
        }
    }
    Ok(())
}

fn staged_sources(root: &Path) -> Result<Vec<(String, String)>, String> {
    let listed = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z", "--", "src/learn"])
        .output()
        .map_err(|e| format!("git ls-files: {e}"))?;
    if !listed.status.success() {
        return Err(format!(
            "git ls-files: {}",
            String::from_utf8_lossy(&listed.stderr).trim()
        ));
    }
    let mut out = Vec::new();
    for raw in listed.stdout.split(|b| *b == 0) {
        if raw.is_empty() {
            continue;
        }
        let path = String::from_utf8(raw.to_vec()).map_err(|e| format!("git path: {e}"))?;
        let blob = Command::new("git")
            .arg("-C")
            .arg(root)
            .arg("show")
            .arg(format!(":{path}"))
            .output()
            .map_err(|e| format!("git show {path}: {e}"))?;
        if !blob.status.success() {
            return Err(format!(
                "git show {path}: {}",
                String::from_utf8_lossy(&blob.stderr).trim()
            ));
        }
        let src = String::from_utf8(blob.stdout).map_err(|e| format!("{path}: not UTF-8: {e}"))?;
        out.push((path, src));
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_names_follow_the_staged_rule() {
        assert!(is_product("gemm_nn"));
        assert!(is_product("row_group_dots"));
        assert!(is_product("wsyrk"));
        assert!(!is_product("matmat_q8act"));
        assert!(!is_product("quant_act"));
    }

    #[test]
    fn a_scalar_product_is_unrooted_and_a_limb_call_is_rooted() {
        let plain = "pub fn dot_probe(a: &[i64], b: &[i64]) -> i64 { \
                     let mut s = 0; for i in 0..a.len() { s += a[i] * b[i]; } s }";
        let items = scan_items(plain).unwrap();
        assert_eq!(items.len(), 1);
        let body = items[0].body.as_deref().unwrap();
        assert!(!rooted(body, false));

        let via_limb = "use crate::engine::kern::limb::gemm_q32;\n\
                        pub fn gemm_probe(a: &[i64], b: &[i64]) -> i64 { \
                        gemm_q32(a, 1, b, 1, 1, 3).unwrap()[0] }";
        let items = scan_items(via_limb).unwrap();
        let body = items[0].body.as_deref().unwrap();
        assert!(rooted(body, true));
    }

    #[test]
    fn a_new_row_is_refused_and_a_fixed_row_is_stale() {
        let base: BTreeSet<(String, String)> = [("src/learn/a.rs".into(), "dot_a".into())].into();
        let current: BTreeSet<(String, String)> = [
            ("src/learn/a.rs".into(), "dot_a".into()),
            ("src/learn/b.rs".into(), "dot_b".into()),
        ]
        .into();
        assert_eq!(current.difference(&base).count(), 1);
        assert_eq!(base.difference(&current).count(), 0);
        let fixed: BTreeSet<(String, String)> = BTreeSet::new();
        assert_eq!(base.difference(&fixed).count(), 1);
    }
}
