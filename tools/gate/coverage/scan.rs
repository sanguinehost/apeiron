//! Lexical scan of a Rust source for its non-test `fn` items.
//!
//! Comments and string/char literals are blanked first so their braces do not move the depth; then
//! a token walk keeps a frame stack (module, `impl`/`trait` with its type, test, other). A `fn` is an
//! item only when the innermost frame is a module, an `impl`/`trait`, or the file root. An unbalanced
//! file is an error: the scanner never reports a count it could not close.

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Item {
    pub name: String,
    pub ty: Option<String>,
    /// Inline `mod` names, outermost first, between the file's module and the item.
    pub mods: Vec<String>,
    pub line: usize,
    /// The function body, braces included, with comments and string/char literals blanked. `None`
    /// for a trait method declaration (`fn f(..);`).
    pub body: Option<String>,
}

#[derive(Clone, PartialEq)]
enum Frame {
    Mod(Option<String>),
    Impl(Option<String>),
    Test,
    Other,
}

/// Blank comments, strings and char literals with spaces, keeping newlines.
fn blank(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut o = b.clone();
    let n = b.len();
    let mut i = 0;
    let wipe = |o: &mut Vec<char>, a: usize, z: usize| {
        for c in o.iter_mut().take(z).skip(a) {
            if *c != '\n' {
                *c = ' ';
            }
        }
    };
    while i < n {
        let c = b[i];
        if c == '/' && i + 1 < n && b[i + 1] == '/' {
            let s = i;
            while i < n && b[i] != '\n' {
                i += 1;
            }
            wipe(&mut o, s, i);
        } else if c == '/' && i + 1 < n && b[i + 1] == '*' {
            let s = i;
            let mut d = 0;
            while i < n {
                if b[i] == '/' && i + 1 < n && b[i + 1] == '*' {
                    d += 1;
                    i += 2;
                } else if b[i] == '*' && i + 1 < n && b[i + 1] == '/' {
                    d -= 1;
                    i += 2;
                    if d == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            wipe(&mut o, s, i);
        } else if c == 'r'
            && i + 1 < n
            && (b[i + 1] == '"' || b[i + 1] == '#')
            && (i == 0 || !(b[i - 1].is_alphanumeric() || b[i - 1] == '_'))
        {
            let mut j = i + 1;
            let mut h = 0;
            while j < n && b[j] == '#' {
                h += 1;
                j += 1;
            }
            if j < n && b[j] == '"' {
                j += 1;
                loop {
                    if j >= n {
                        break;
                    }
                    if b[j] == '"' && (0..h).all(|k| j + 1 + k < n && b[j + 1 + k] == '#') {
                        j += 1 + h;
                        break;
                    }
                    j += 1;
                }
                wipe(&mut o, i, j);
                i = j;
            } else {
                i += 1;
            }
        } else if c == '"' {
            let s = i;
            i += 1;
            while i < n && b[i] != '"' {
                i += if b[i] == '\\' { 2 } else { 1 };
            }
            i = (i + 1).min(n);
            wipe(&mut o, s, i);
        } else if c == '\'' {
            // char literal `'x'` / `'\n'` / `'\u{..}'`; otherwise a lifetime.
            if i + 2 < n && b[i + 1] == '\\' {
                let mut j = i + 2;
                while j < n && b[j] != '\'' {
                    j += 1;
                }
                wipe(&mut o, i, (j + 1).min(n));
                i = (j + 1).min(n);
            } else if i + 2 < n && b[i + 2] == '\'' {
                wipe(&mut o, i, i + 3);
                i += 3;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    o.into_iter().collect()
}

/// Cost: time O(L) in source characters, space O(L + depth).
pub(crate) fn scan_items(src: &str) -> Result<Vec<Item>, String> {
    let code = blank(src);
    let ch: Vec<char> = code.chars().collect();
    let n = ch.len();
    let mut items = Vec::new();
    let mut stack: Vec<Frame> = Vec::new();
    // Parallel to `stack`: for the frame that is a fn body, the `{` position and the item index.
    let mut marks: Vec<Option<(usize, usize)>> = Vec::new();
    let mut pending_test = false;
    let mut line = 1usize;
    let mut i = 0;

    let item_level =
        |s: &Vec<Frame>| matches!(s.last(), None | Some(Frame::Mod(_)) | Some(Frame::Impl(_)));
    let ident = |ch: &Vec<char>, i: usize| -> Option<(String, usize)> {
        let mut j = i;
        while j < ch.len() && (ch[j].is_alphanumeric() || ch[j] == '_') {
            j += 1;
        }
        (j > i).then(|| (ch[i..j].iter().collect(), j))
    };
    let skip_ws = |ch: &Vec<char>, mut i: usize, line: &mut usize| {
        while i < ch.len() && ch[i].is_whitespace() {
            if ch[i] == '\n' {
                *line += 1;
            }
            i += 1;
        }
        i
    };

    while i < n {
        let c = ch[i];
        if c == '\n' {
            line += 1;
            i += 1;
        } else if c == '#' {
            // attribute: `#[…]` or `#![…]`
            let mut j = i + 1;
            if j < n && ch[j] == '!' {
                j += 1;
            }
            if j < n && ch[j] == '[' {
                let mut d = 0;
                let s = j;
                while j < n {
                    if ch[j] == '[' {
                        d += 1;
                    } else if ch[j] == ']' {
                        d -= 1;
                        if d == 0 {
                            break;
                        }
                    } else if ch[j] == '\n' {
                        line += 1;
                    }
                    j += 1;
                }
                let attr: String = ch[s + 1..j.min(n)]
                    .iter()
                    .filter(|c| !c.is_whitespace())
                    .collect();
                let inner = ch[i + 1] == '!';
                if !inner
                    && (attr == "test"
                        || attr.starts_with("cfg(test")
                        || attr.starts_with("cfg(all(test")
                        || attr.starts_with("tokio::test"))
                {
                    pending_test = true;
                }
                i = j + 1;
            } else {
                i += 1;
            }
        } else if c.is_alphabetic() || c == '_' {
            let (w, j) = ident(&ch, i).unwrap();
            let prev_ok =
                i == 0 || !(ch[i - 1] == '.' || ch[i - 1] == ':' && i >= 2 && ch[i - 2] == ':');
            match w.as_str() {
                "fn" if prev_ok => {
                    let k = skip_ws(&ch, j, &mut line);
                    if let Some((name, k2)) = ident(&ch, k) {
                        // the signature runs to the first `{` or `;` outside parens/brackets
                        let (mut p, mut m) = (0i32, k2);
                        while m < n {
                            match ch[m] {
                                '(' | '[' => p += 1,
                                ')' | ']' => p -= 1,
                                '{' | ';' if p == 0 => break,
                                '\n' => line += 1,
                                _ => {}
                            }
                            m += 1;
                        }
                        if m >= n {
                            return Err(format!("fn `{name}` has no body or `;`"));
                        }
                        let item = if item_level(&stack) && !pending_test {
                            let ty = match stack.last() {
                                Some(Frame::Impl(t)) => t.clone(),
                                _ => None,
                            };
                            let mods = stack
                                .iter()
                                .filter_map(|f| match f {
                                    Frame::Mod(Some(n)) => Some(n.clone()),
                                    _ => None,
                                })
                                .collect();
                            items.push(Item {
                                name,
                                ty,
                                mods,
                                line,
                                body: None,
                            });
                            Some(items.len() - 1)
                        } else {
                            None
                        };
                        if ch[m] == '{' {
                            stack.push(if pending_test {
                                Frame::Test
                            } else {
                                Frame::Other
                            });
                            marks.push(item.map(|idx| (m, idx)));
                        }
                        pending_test = false;
                        i = m + 1;
                    } else {
                        i = j;
                    }
                }
                "mod" if prev_ok => {
                    let k = skip_ws(&ch, j, &mut line);
                    let (name, k2) = ident(&ch, k).unwrap_or((String::new(), k));
                    let k3 = skip_ws(&ch, k2, &mut line);
                    if k3 < n && ch[k3] == '{' {
                        stack.push(if pending_test || !item_level(&stack) {
                            Frame::Test
                        } else {
                            Frame::Mod(Some(name))
                        });
                        marks.push(None);
                        i = k3 + 1;
                    } else {
                        i = k3;
                    }
                    pending_test = false;
                }
                "impl" | "trait" if prev_ok && item_level(&stack) => {
                    let (mut m, mut angle) = (j, 0i32);
                    let mut head = String::new();
                    while m < n && !(ch[m] == '{' && angle == 0) && ch[m] != ';' {
                        match ch[m] {
                            '<' => angle += 1,
                            '>' if m > 0 && ch[m - 1] != '-' => angle -= 1,
                            '\n' => line += 1,
                            _ => {}
                        }
                        if angle == 0 && ch[m] != '<' && ch[m] != '>' {
                            head.push(ch[m]);
                        } else if angle == 0 {
                            head.push(' ');
                        }
                        m += 1;
                    }
                    if m >= n {
                        return Err("impl without a body".into());
                    }
                    if ch[m] == ';' {
                        i = m + 1;
                    } else {
                        let words: Vec<&str> = head
                            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                            .filter(|w| !w.is_empty())
                            .collect();
                        let ty = if w == "trait" {
                            words.first().map(|s| s.to_string())
                        } else if let Some(p) = words.iter().rposition(|w| *w == "for") {
                            words.get(p + 1).map(|s| s.to_string())
                        } else {
                            words
                                .iter()
                                .find(|w| !matches!(**w, "unsafe" | "const" | "dyn"))
                                .map(|s| s.to_string())
                        };
                        stack.push(if pending_test {
                            Frame::Test
                        } else {
                            Frame::Impl(ty)
                        });
                        marks.push(None);
                        i = m + 1;
                    }
                    pending_test = false;
                }
                _ => i = j,
            }
        } else if c == '{' {
            stack.push(if pending_test {
                Frame::Test
            } else {
                Frame::Other
            });
            marks.push(None);
            pending_test = false;
            i += 1;
        } else if c == '}' {
            if stack.pop().is_none() {
                return Err(format!("unmatched `}}` at line {line}"));
            }
            if let Some(Some((start, idx))) = marks.pop() {
                if let Some(item) = items.get_mut(idx) {
                    item.body = Some(ch[start..=i].iter().collect());
                }
            }
            i += 1;
        } else if c == ';' {
            pending_test = false;
            i += 1;
        } else {
            i += 1;
        }
    }
    if !stack.is_empty() {
        return Err(format!("{} unclosed block(s) at end of file", stack.len()));
    }
    Ok(items)
}
