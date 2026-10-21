//! Clone detection over normalised token sequences.
//!
//! A body is flattened to its tokens; a called or accessed name (an ident before `(` or `!`, after
//! `.`, or beside `::`) and every keyword stay verbatim, every other ident becomes `$k` numbered by
//! first occurrence, and every literal becomes `$n` or `$s`. Renaming locals, reordering nothing,
//! changing constants: the sequence does not move. Two bodies are one group when their sequences
//! are equal (≥ 40 tokens) or their 8-gram shingle sets have Jaccard similarity ≥ 0.9 (both ≥ 80
//! tokens).

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

use proc_macro2::{Delimiter, Spacing, TokenStream, TokenTree};
use quote::ToTokens;

use super::load::Tree;

const EXACT_MIN: usize = 40;
const NEAR_MIN: usize = 80;
const SHINGLE: usize = 8;
/// A shingle present in more fns than this is boilerplate and indexes nothing.
const POSTING_MAX: usize = 50;
/// Jaccard ≥ 9/10, compared in integers.
const JACCARD: (usize, usize) = (9, 10);

#[derive(Clone, PartialEq)]
enum Raw {
    Ident(String),
    Punct(char, bool),
    /// The literal's text, and whether it is a string.
    Lit(String, bool),
    Open(char),
    Close(char),
}

fn flatten(ts: TokenStream, out: &mut Vec<Raw>) {
    for tt in ts {
        match tt {
            TokenTree::Ident(i) => out.push(Raw::Ident(i.to_string())),
            TokenTree::Punct(p) => out.push(Raw::Punct(p.as_char(), p.spacing() == Spacing::Joint)),
            TokenTree::Literal(l) => {
                let t = l.to_string();
                let s = t.contains('"');
                out.push(Raw::Lit(t, s));
            }
            TokenTree::Group(g) => {
                let (o, c) = match g.delimiter() {
                    Delimiter::Parenthesis => ('(', ')'),
                    Delimiter::Brace => ('{', '}'),
                    Delimiter::Bracket => ('[', ']'),
                    Delimiter::None => (' ', ' '),
                };
                out.push(Raw::Open(o));
                flatten(g.stream(), out);
                out.push(Raw::Close(c));
            }
        }
    }
}

const KEYWORDS: [&str; 38] = [
    "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern",
    "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub",
    "ref", "return", "self", "Self", "static", "struct", "super", "trait", "true", "type",
    "unsafe", "use", "where", "while",
];

fn is_colons(raw: &[Raw], i: usize) -> bool {
    matches!(
        (raw.get(i), raw.get(i + 1)),
        (Some(Raw::Punct(':', true)), Some(Raw::Punct(':', _)))
    )
}

/// The normalised token sequence of a body, each token hashed. A `const`/`static` initializer
/// (`data`) is a table, not code: its literals stay verbatim, so two tables are one group only
/// when their values are equal.
///
/// Cost: time and space O(T) over the T tokens of the body.
pub(crate) fn normalise(body: &syn::Block, data: bool) -> Vec<u64> {
    let mut raw = Vec::new();
    flatten(body.to_token_stream(), &mut raw);
    let mut names: BTreeMap<String, usize> = BTreeMap::new();
    let mut out = Vec::with_capacity(raw.len());
    for (i, t) in raw.iter().enumerate() {
        let s = match t {
            Raw::Ident(id) => {
                let called = matches!(
                    raw.get(i + 1),
                    Some(Raw::Open('(')) | Some(Raw::Punct('!', _))
                );
                let after_dot = i > 0 && matches!(raw[i - 1], Raw::Punct('.', _));
                let pathy = is_colons(&raw, i + 1) || (i >= 2 && is_colons(&raw, i - 2));
                if called || after_dot || pathy || KEYWORDS.contains(&id.as_str()) {
                    id.clone()
                } else {
                    let n = names.len();
                    format!("${}", names.entry(id.clone()).or_insert(n))
                }
            }
            Raw::Punct(c, _) => c.to_string(),
            Raw::Lit(t, _) if data => t.clone(),
            Raw::Lit(_, true) => "$s".into(),
            Raw::Lit(_, false) => "$n".into(),
            Raw::Open(c) | Raw::Close(c) => c.to_string(),
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut h);
        out.push(h.finish());
    }
    out
}

fn find(parent: &mut [usize], mut v: usize) -> usize {
    while parent[v] != v {
        parent[v] = parent[parent[v]];
        v = parent[v];
    }
    v
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra.max(rb)] = ra.min(rb);
    }
}

/// The clone groups that touch non-test `src/`: `clone` when two or more members are non-test
/// `src/` fns, `clone-test` when one is and the rest are tests (a test restating the body it tests).
///
/// Cost: time O(T + Σ_s p_s²) over T tokens and the posting length p_s ≤ 50 of each shingle s;
/// space O(T).
pub(crate) fn groups(tree: &Tree) -> Vec<(Vec<usize>, &'static str)> {
    let n = tree.fns.len();
    let seqs: Vec<Vec<u64>> = tree.fns.iter().map(|f| normalise(&f.body, f.init)).collect();
    let mut parent: Vec<usize> = (0..n).collect();

    let mut exact: BTreeMap<&[u64], usize> = BTreeMap::new();
    for (i, s) in seqs.iter().enumerate() {
        if s.len() >= EXACT_MIN {
            match exact.get(s.as_slice()) {
                Some(&j) => union(&mut parent, i, j),
                None => {
                    exact.insert(s.as_slice(), i);
                }
            }
        }
    }

    let shingles: Vec<BTreeSet<u64>> = seqs
        .iter()
        .map(|s| {
            if s.len() < NEAR_MIN {
                return BTreeSet::new();
            }
            s.windows(SHINGLE)
                .map(|w| {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    w.hash(&mut h);
                    h.finish()
                })
                .collect()
        })
        .collect();
    let mut posting: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (i, set) in shingles.iter().enumerate() {
        for &h in set {
            posting.entry(h).or_default().push(i);
        }
    }
    let mut shared: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for list in posting.values() {
        if list.len() < 2 || list.len() > POSTING_MAX {
            continue;
        }
        for (a, &i) in list.iter().enumerate() {
            for &j in &list[a + 1..] {
                *shared.entry((i, j)).or_insert(0) += 1;
            }
        }
    }
    for (&(i, j), &inter) in &shared {
        let union_size = shingles[i].len() + shingles[j].len() - inter;
        if inter * JACCARD.1 >= union_size * JACCARD.0 {
            union(&mut parent, i, j);
        }
    }

    let mut by_root: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        by_root.entry(find(&mut parent, i)).or_default().push(i);
    }
    let mut out = Vec::new();
    for (_, members) in by_root {
        if members.len() < 2 {
            continue;
        }
        let src = members
            .iter()
            .filter(|&&i| tree.fns[i].file.starts_with("src/") && !tree.fns[i].test)
            .count();
        let kind = match src {
            0 => continue,
            1 => "clone-test",
            _ => "clone",
        };
        out.push((members, kind));
    }
    out
}
